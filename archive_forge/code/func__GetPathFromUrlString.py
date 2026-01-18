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
def _GetPathFromUrlString(url_str):
    """Returns path component of a URL string."""
    end_scheme_idx = url_str.find('://')
    if end_scheme_idx == -1:
        return url_str
    else:
        return url_str[end_scheme_idx + 3:]