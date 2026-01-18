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
def _ExcludeDir(self, dir):
    """Check a directory to see if it should be excluded from os.walk.

    Args:
      dir: String representing the directory to check.

    Returns:
      True if the directory should be excluded.
    """
    if self.exclude_tuple is None:
        return False
    base_url, exclude_dirs, exclude_pattern = self.exclude_tuple
    if not exclude_dirs:
        return False
    str_to_check = StorageUrlFromString(dir).url_string[len(base_url.url_string):]
    if str_to_check.startswith(self.wildcard_url.delim):
        str_to_check = str_to_check[1:]
    if exclude_pattern.match(str_to_check):
        if self.logger:
            self.logger.info('Skipping excluded directory %s...', dir)
        return True