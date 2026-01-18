from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import glob
import logging
import os
import re
import textwrap
import six
from gslib.bucket_listing_ref import BucketListingBucket
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import NotFoundException
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.storage_url import WILDCARD_REGEX
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.text_util import FixWindowsEncodingIfNeeded
from gslib.utils.text_util import PrintableStr
class CloudWildcardIterator(WildcardIterator):
    """WildcardIterator subclass for buckets, bucket subdirs and objects.

  Iterates over BucketListingRef matching the Url string wildcard. It's
  much more efficient to first get metadata that's available in the Bucket
  (for example to get the name and size of each object), because that
  information is available in the object list results.
  """

    def __init__(self, wildcard_url, gsutil_api, all_versions=False, project_id=None, logger=None):
        """Instantiates an iterator that matches the wildcard URL.

    Args:
      wildcard_url: CloudUrl that contains the wildcard to iterate.
      gsutil_api: Cloud storage interface.  Passed in for thread safety, also
                  settable for testing/mocking.
      all_versions: If true, the iterator yields all versions of objects
                    matching the wildcard.  If false, yields just the live
                    object version.
      project_id: Project ID to use for bucket listings.
      logger: logging.Logger used for outputting debug messages during
              iteration. If None, the root logger will be used.
    """
        self.wildcard_url = wildcard_url
        self.all_versions = all_versions
        self.gsutil_api = gsutil_api
        self.project_id = project_id
        self.logger = logger or logging.getLogger()

    def __iter__(self, bucket_listing_fields=None, expand_top_level_buckets=False):
        """Iterator that gets called when iterating over the cloud wildcard.

    In the case where no wildcard is present, returns a single matching object,
    single matching prefix, or one of each if both exist.

    Args:
      bucket_listing_fields: Iterable fields to include in bucket listings.
                             Ex. ['name', 'acl'].  Iterator is
                             responsible for converting these to list-style
                             format ['items/name', 'items/acl'] as well as
                             adding any fields necessary for listing such as
                             prefixes.  API implementation is responsible for
                             adding pagination fields.  If this is None,
                             all fields are returned.
      expand_top_level_buckets: If true, yield no BUCKET references.  Instead,
                                expand buckets into top-level objects and
                                prefixes.

    Yields:
      BucketListingRef of type BUCKET, OBJECT or PREFIX.
    """
        single_version_request = self.wildcard_url.HasGeneration()
        get_fields = None
        if bucket_listing_fields:
            get_fields = set()
            for field in bucket_listing_fields:
                get_fields.add(field)
            bucket_listing_fields = self._GetToListFields(get_fields=bucket_listing_fields)
            bucket_listing_fields.update(['items/name', 'prefixes'])
            get_fields.update(['name'])
            if single_version_request or self.all_versions:
                bucket_listing_fields.update(['items/generation', 'items/metageneration'])
                get_fields.update(['generation', 'metageneration'])
        for bucket_listing_ref in self._ExpandBucketWildcards(bucket_fields=['id']):
            bucket_url_string = bucket_listing_ref.url_string
            if self.wildcard_url.IsBucket():
                if expand_top_level_buckets:
                    url = StorageUrlFromString(bucket_url_string)
                    for obj_or_prefix in self.gsutil_api.ListObjects(url.bucket_name, delimiter='/', all_versions=self.all_versions, provider=self.wildcard_url.scheme, fields=bucket_listing_fields):
                        if obj_or_prefix.datatype == CloudApi.CsObjectOrPrefixType.OBJECT:
                            yield self._GetObjectRef(bucket_url_string, obj_or_prefix.data, with_version=self.all_versions)
                        else:
                            yield self._GetPrefixRef(bucket_url_string, obj_or_prefix.data)
                else:
                    yield bucket_listing_ref
            else:
                if not ContainsWildcard(self.wildcard_url.url_string) and self.wildcard_url.IsObject() and (not self.all_versions):
                    try:
                        get_object = self.gsutil_api.GetObjectMetadata(self.wildcard_url.bucket_name, self.wildcard_url.object_name, generation=self.wildcard_url.generation, provider=self.wildcard_url.scheme, fields=get_fields)
                        yield self._GetObjectRef(self.wildcard_url.bucket_url_string, get_object, with_version=self.all_versions or single_version_request)
                        return
                    except (NotFoundException, AccessDeniedException):
                        pass
                if single_version_request:
                    url_string = '%s%s#%s' % (bucket_url_string, self.wildcard_url.object_name, self.wildcard_url.generation)
                else:
                    url_string = '%s%s' % (bucket_url_string, StripOneSlash(self.wildcard_url.object_name) or '/')
                urls_needing_expansion = [url_string]
                while urls_needing_expansion:
                    url = StorageUrlFromString(urls_needing_expansion.pop(0))
                    prefix, delimiter, prefix_wildcard, suffix_wildcard = self._BuildBucketFilterStrings(url.object_name)
                    regex_patterns = self._GetRegexPatterns(prefix_wildcard)
                    listing_fields = set(['prefixes']) if suffix_wildcard else bucket_listing_fields
                    for obj_or_prefix in self.gsutil_api.ListObjects(url.bucket_name, prefix=prefix, delimiter=delimiter, all_versions=self.all_versions or single_version_request, provider=self.wildcard_url.scheme, fields=listing_fields):
                        for pattern in regex_patterns:
                            if obj_or_prefix.datatype == CloudApi.CsObjectOrPrefixType.OBJECT:
                                gcs_object = obj_or_prefix.data
                                if pattern.match(gcs_object.name):
                                    if not suffix_wildcard or StripOneSlash(gcs_object.name) == suffix_wildcard:
                                        if not single_version_request or self._SingleVersionMatches(gcs_object.generation):
                                            yield self._GetObjectRef(bucket_url_string, gcs_object, with_version=self.all_versions or single_version_request)
                                    break
                            else:
                                prefix = obj_or_prefix.data
                                if ContainsWildcard(prefix):
                                    raise CommandException('Cloud folder %s%s contains a wildcard; gsutil does not currently support objects with wildcards in their name.' % (bucket_url_string, prefix))
                                rstripped_prefix = StripOneSlash(prefix)
                                if pattern.match(rstripped_prefix):
                                    if suffix_wildcard and rstripped_prefix != suffix_wildcard:
                                        url_append_string = '%s%s' % (bucket_url_string, rstripped_prefix + '/' + suffix_wildcard)
                                        urls_needing_expansion.append(url_append_string)
                                    else:
                                        yield self._GetPrefixRef(bucket_url_string, prefix)
                                    break

    def _GetRegexPatterns(self, wildcard_pattern):
        """Returns list of regex patterns derived from the wildcard patterns.

    Args:
      wildcard_pattern (str): A wilcard_pattern to filter the resources.

    Returns:
      List of compiled regex patterns.

    This translates the wildcard_pattern and also creates some additional
    patterns so that we can treat ** in a/b/c/**/d.txt as zero or more folders.
    This means, a/b/c/d.txt will also be returned along with a/b/c/e/f/d.txt.
    """
        wildcard_patterns = [wildcard_pattern]
        if '/**/' in wildcard_pattern:
            updated_pattern = wildcard_pattern.replace('/**/', '/')
            wildcard_patterns.append(updated_pattern)
        else:
            updated_pattern = wildcard_pattern
        for pattern in (wildcard_pattern, updated_pattern):
            if pattern.startswith('**/'):
                wildcard_patterns.append(pattern[3:])
        return [re.compile(fnmatch.translate(p)) for p in wildcard_patterns]

    def _BuildBucketFilterStrings(self, wildcard):
        """Builds strings needed for querying a bucket and filtering results.

    This implements wildcard object name matching.

    Args:
      wildcard: The wildcard string to match to objects.

    Returns:
      (prefix, delimiter, prefix_wildcard, suffix_wildcard)
      where:
        prefix is the prefix to be sent in bucket GET request.
        delimiter is the delimiter to be sent in bucket GET request.
        prefix_wildcard is the wildcard to be used to filter bucket GET results.
        suffix_wildcard is wildcard to be appended to filtered bucket GET
          results for next wildcard expansion iteration.
      For example, given the wildcard gs://bucket/abc/d*e/f*.txt we
      would build prefix= abc/d, delimiter=/, prefix_wildcard=d*e, and
      suffix_wildcard=f*.txt. Using this prefix and delimiter for a bucket
      listing request will then produce a listing result set that can be
      filtered using this prefix_wildcard; and we'd use this suffix_wildcard
      to feed into the next call(s) to _BuildBucketFilterStrings(), for the
      next iteration of listing/filtering.

    Raises:
      AssertionError if wildcard doesn't contain any wildcard chars.
    """
        match = WILDCARD_REGEX.search(wildcard)
        if not match:
            prefix = wildcard
            delimiter = '/'
            prefix_wildcard = wildcard
            suffix_wildcard = ''
        else:
            if match.start() > 0:
                prefix = wildcard[:match.start()]
                wildcard_part = wildcard[match.start():]
            else:
                prefix = None
                wildcard_part = wildcard
            end = wildcard_part.find('/')
            if end != -1:
                wildcard_part = wildcard_part[:end + 1]
            prefix_wildcard = (prefix or '') + wildcard_part
            if not prefix_wildcard.endswith('**/'):
                prefix_wildcard = StripOneSlash(prefix_wildcard)
            suffix_wildcard = wildcard[match.end():]
            end = suffix_wildcard.find('/')
            if end == -1:
                suffix_wildcard = ''
            else:
                suffix_wildcard = suffix_wildcard[end + 1:]
            if prefix_wildcard.find('**') != -1:
                delimiter = None
                prefix_wildcard += suffix_wildcard
                suffix_wildcard = ''
            else:
                delimiter = '/'
        self.logger.debug('wildcard=%s, prefix=%s, delimiter=%s, prefix_wildcard=%s, suffix_wildcard=%s\n', PrintableStr(wildcard), PrintableStr(prefix), PrintableStr(delimiter), PrintableStr(prefix_wildcard), PrintableStr(suffix_wildcard))
        return (prefix, delimiter, prefix_wildcard, suffix_wildcard)

    def _SingleVersionMatches(self, listed_generation):
        decoded_generation = GenerationFromUrlAndString(self.wildcard_url, listed_generation)
        return str(self.wildcard_url.generation) == str(decoded_generation)

    def _ExpandBucketWildcards(self, bucket_fields=None):
        """Expands bucket and provider wildcards.

    Builds a list of bucket url strings that can be iterated on.

    Args:
      bucket_fields: If present, populate only these metadata fields for
                     buckets.  Example value: ['acl', 'defaultObjectAcl']

    Yields:
      BucketListingRefereneces of type BUCKET.
    """
        bucket_url = StorageUrlFromString(self.wildcard_url.bucket_url_string)
        if bucket_fields and set(bucket_fields) == set(['id']) and (not ContainsWildcard(self.wildcard_url.bucket_name)):
            yield BucketListingBucket(bucket_url)
        elif self.wildcard_url.IsBucket() and (not ContainsWildcard(self.wildcard_url.bucket_name)):
            yield BucketListingBucket(bucket_url, root_object=self.gsutil_api.GetBucket(self.wildcard_url.bucket_name, provider=self.wildcard_url.scheme, fields=bucket_fields))
        else:
            regex = fnmatch.translate(self.wildcard_url.bucket_name)
            prog = re.compile(regex)
            fields = self._GetToListFields(bucket_fields)
            if fields:
                fields.add('items/id')
            for bucket in self.gsutil_api.ListBuckets(fields=fields, project_id=self.project_id, provider=self.wildcard_url.scheme):
                if prog.match(bucket.id):
                    url = StorageUrlFromString('%s://%s/' % (self.wildcard_url.scheme, bucket.id))
                    yield BucketListingBucket(url, root_object=bucket)

    def _GetToListFields(self, get_fields=None):
        """Prepends 'items/' to the input fields and converts it to a set.

    This way field sets requested for GetBucket can be used in ListBucket calls.
    Note that the input set must contain only bucket or object fields; listing
    fields such as prefixes or nextPageToken should be added after calling
    this function.

    Args:
      get_fields: Iterable fields usable in GetBucket/GetObject calls.

    Returns:
      Set of fields usable in ListBuckets/ListObjects calls.
    """
        if get_fields:
            list_fields = set()
            for field in get_fields:
                list_fields.add('items/' + field)
            return list_fields

    def _GetObjectRef(self, bucket_url_string, gcs_object, with_version=False):
        """Creates a BucketListingRef of type OBJECT from the arguments.

    Args:
      bucket_url_string: Wildcardless string describing the containing bucket.
      gcs_object: gsutil_api root Object for populating the BucketListingRef.
      with_version: If true, return a reference with a versioned string.

    Returns:
      BucketListingRef of type OBJECT.
    """
        if with_version and gcs_object.generation is not None:
            generation_str = GenerationFromUrlAndString(self.wildcard_url, gcs_object.generation)
            object_string = '%s%s#%s' % (bucket_url_string, gcs_object.name, generation_str)
        else:
            object_string = '%s%s' % (bucket_url_string, gcs_object.name)
        object_url = StorageUrlFromString(object_string)
        return BucketListingObject(object_url, root_object=gcs_object)

    def _GetPrefixRef(self, bucket_url_string, prefix):
        """Creates a BucketListingRef of type PREFIX from the arguments.

    Args:
      bucket_url_string: Wildcardless string describing the containing bucket.
      prefix: gsutil_api Prefix for populating the BucketListingRef

    Returns:
      BucketListingRef of type PREFIX.
    """
        prefix_url = StorageUrlFromString('%s%s' % (bucket_url_string, prefix))
        return BucketListingPrefix(prefix_url, root_object=prefix)

    def IterBuckets(self, bucket_fields=None):
        """Iterates over the wildcard, returning refs for each expanded bucket.

    This ignores the object part of the URL entirely and expands only the
    the bucket portion.  It will yield BucketListingRefs of type BUCKET only.

    Args:
      bucket_fields: Iterable fields to include in bucket listings.
                     Ex. ['defaultObjectAcl', 'logging'].  This function is
                     responsible for converting these to listing-style
                     format ['items/defaultObjectAcl', 'items/logging'], as
                     well as adding any fields necessary for listing such as
                     'items/id'.  API implemenation is responsible for
                     adding pagination fields.  If this is None, all fields are
                     returned.

    Yields:
      BucketListingRef of type BUCKET, or empty iterator if no matches.
    """
        for blr in self._ExpandBucketWildcards(bucket_fields=bucket_fields):
            yield blr

    def IterAll(self, bucket_listing_fields=None, expand_top_level_buckets=False):
        """Iterates over the wildcard, yielding bucket, prefix or object refs.

    Args:
      bucket_listing_fields: If present, populate only these metadata
                             fields for listed objects.
      expand_top_level_buckets: If true and the wildcard expands only to
                                Bucket(s), yields the expansion of each bucket
                                into a top-level listing of prefixes and objects
                                in that bucket instead of a BucketListingRef
                                to that bucket.

    Yields:
      BucketListingRef, or empty iterator if no matches.
    """
        for blr in self.__iter__(bucket_listing_fields=bucket_listing_fields, expand_top_level_buckets=expand_top_level_buckets):
            yield blr

    def IterObjects(self, bucket_listing_fields=None):
        """Iterates over the wildcard, yielding only object BucketListingRefs.

    Args:
      bucket_listing_fields: If present, populate only these metadata
                             fields for listed objects.

    Yields:
      BucketListingRefs of type OBJECT or empty iterator if no matches.
    """
        for blr in self.__iter__(bucket_listing_fields=bucket_listing_fields, expand_top_level_buckets=True):
            if blr.IsObject():
                yield blr