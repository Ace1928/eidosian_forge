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
def HaveProviderUrls(args_to_check):
    """Checks whether args_to_check contains any provider URLs (like 'gs://').

  Args:
    args_to_check: Command-line argument subset to check.

  Returns:
    True if args_to_check contains any provider URLs.
  """
    for url_str in args_to_check:
        storage_url = StorageUrlFromString(url_str)
        if storage_url.IsCloudUrl() and storage_url.IsProvider():
            return True
    return False