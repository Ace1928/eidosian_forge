from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import binascii
import codecs
import os
import sys
import io
import re
import locale
import collections
import random
import six
import string
from six.moves import urllib
from six.moves import range
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.utils.constants import UTF8
from gslib.utils.constants import WINDOWS_1252
from gslib.utils.system_util import IS_CP1252
def InsistAsciiHeader(header):
    """Checks for ASCII-only characters in `header`.

    Also constructs an error message using `header` if the check fails.

    Args:
      header: Union[str, binary, unicode] Text being checked for ASCII values.

    Returns:
      None
    """
    InsistAscii(header, 'Invalid non-ASCII header (%s).' % header)