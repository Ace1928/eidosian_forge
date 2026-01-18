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