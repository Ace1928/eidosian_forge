from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def IsCloudSubdirPlaceholder(url, blr=None):
    """Determines if a StorageUrl is a cloud subdir placeholder.

  This function is needed because GUI tools (like the GCS cloud console) allow
  users to create empty "folders" by creating a placeholder object; and parts
  of gsutil need to treat those placeholder objects specially. For example,
  gsutil rsync needs to avoid downloading those objects because they can cause
  conflicts (see comments in rsync command for details).

  We currently detect two cases:
    - Cloud objects whose name ends with '_$folder$'
    - Cloud objects whose name ends with '/'

  Args:
    url: (gslib.storage_url.StorageUrl) The URL to be checked.
    blr: (gslib.BucketListingRef or None) The blr to check, or None if not
        available. If `blr` is None, size won't be checked.

  Returns:
    (bool) True if the URL is a cloud subdir placeholder, otherwise False.
  """
    if not url.IsCloudUrl():
        return False
    url_str = url.url_string
    if url_str.endswith('_$folder$'):
        return True
    if blr and blr.IsObject():
        size = blr.root_object.size
    else:
        size = 0
    return size == 0 and url_str.endswith('/')