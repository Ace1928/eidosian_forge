from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import sys
import six
from gslib.cloud_api import EncryptionException
from gslib.exception import CommandException
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GenerationFromUrlAndString
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import text_util
from gslib.wildcard_iterator import StorageUrlFromString
def ExpandUrlAndPrint(self, url):
    """Iterates over the given URL and calls print functions.

    Args:
      url: StorageUrl to iterate over.

    Returns:
      (num_objects, num_bytes) total number of objects and bytes iterated.
    """
    num_objects = 0
    num_dirs = 0
    num_bytes = 0
    print_newline = False
    if url.IsBucket() or self.should_recurse:
        if url.IsBucket():
            self._print_bucket_header_func(url)
        return self._RecurseExpandUrlAndPrint(url.url_string, print_initial_newline=False)
    else:
        if url.HasGeneration():
            iteration_url = url.url_string
        else:
            iteration_url = url.CreatePrefixUrl()
        top_level_iterator = PluralityCheckableIterator(self._iterator_func(iteration_url, all_versions=self.all_versions).IterAll(expand_top_level_buckets=True, bucket_listing_fields=self.bucket_listing_fields))
        plurality = top_level_iterator.HasPlurality()
        try:
            top_level_iterator.PeekException()
        except EncryptionException:
            top_level_iterator = PluralityCheckableIterator(self._iterator_func(url.CreatePrefixUrl(wildcard_suffix=None), all_versions=self.all_versions).IterAll(expand_top_level_buckets=True, bucket_listing_fields=UNENCRYPTED_FULL_LISTING_FIELDS))
            plurality = top_level_iterator.HasPlurality()
        for blr in top_level_iterator:
            if self._MatchesExcludedPattern(blr):
                continue
            if blr.IsObject():
                nd = 0
                no, nb = self._print_object_func(blr)
                print_newline = True
            elif blr.IsPrefix():
                if print_newline:
                    self._print_newline_func()
                else:
                    print_newline = True
                if plurality and self.list_subdir_contents:
                    self._print_dir_header_func(blr)
                elif plurality and (not self.list_subdir_contents):
                    print_newline = False
                expansion_url_str = StorageUrlFromString(blr.url_string).CreatePrefixUrl(wildcard_suffix='*' if self.list_subdir_contents else None)
                nd, no, nb = self._RecurseExpandUrlAndPrint(expansion_url_str)
                self._print_dir_summary_func(nb, blr)
            else:
                raise CommandException('Sub-level iterator returned a CsBucketListingRef of type Bucket')
            num_objects += no
            num_dirs += nd
            num_bytes += nb
        return (num_dirs, num_objects, num_bytes)