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