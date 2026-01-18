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
def UrlsAreForSingleProvider(url_args):
    """Tests whether the URLs are all for a single provider.

  Args:
    url_args: (Iterable[str]) Collection of strings to check.

  Returns:
    True if all URLs are for single provider; False if `url_args` was empty (as
    this would not result in a single unique provider) or URLs targeted multiple
    unique providers.
  """
    provider = None
    url = None
    for url_str in url_args:
        url = StorageUrlFromString(url_str)
        if not provider:
            provider = url.scheme
        elif url.scheme != provider:
            return False
    return provider is not None