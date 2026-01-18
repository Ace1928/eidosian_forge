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
def _RecurseExpandUrlAndPrint(self, url_str, print_initial_newline=True):
    """Iterates over the given URL string and calls print functions.

    Args:
      url_str: String describing StorageUrl to iterate over.
               Must be of depth one or higher.
      print_initial_newline: If true, print a newline before recursively
                             expanded prefixes.

    Returns:
      (num_objects, num_bytes) total number of objects and bytes iterated.
    """
    num_objects = 0
    num_dirs = 0
    num_bytes = 0
    for blr in self._iterator_func('%s' % url_str, all_versions=self.all_versions).IterAll(expand_top_level_buckets=True, bucket_listing_fields=self.bucket_listing_fields):
        if self._MatchesExcludedPattern(blr):
            continue
        if blr.IsObject():
            nd = 0
            no, nb = self._print_object_func(blr)
        elif blr.IsPrefix():
            if self.should_recurse:
                if print_initial_newline:
                    self._print_newline_func()
                else:
                    print_initial_newline = True
                self._print_dir_header_func(blr)
                expansion_url_str = StorageUrlFromString(blr.url_string).CreatePrefixUrl(wildcard_suffix='*')
                nd, no, nb = self._RecurseExpandUrlAndPrint(expansion_url_str)
                self._print_dir_summary_func(nb, blr)
            else:
                nd, no, nb = (1, 0, 0)
                self._print_dir_func(blr)
        else:
            raise CommandException('Sub-level iterator returned a bucketListingRef of type Bucket')
        num_dirs += nd
        num_objects += no
        num_bytes += nb
    return (num_dirs, num_objects, num_bytes)