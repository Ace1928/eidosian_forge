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