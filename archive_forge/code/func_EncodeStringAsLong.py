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
def EncodeStringAsLong(string_to_convert):
    """Encodes an ASCII string as a python long.

  This is used for modeling S3 version_id's as apitools generation.  Because
  python longs can be arbitrarily large, this works.

  Args:
    string_to_convert: ASCII string to convert to a long.

  Returns:
    Long that represents the input string.
  """
    hex_bytestr = codecs.encode(six.ensure_binary(string_to_convert), 'hex_codec')
    return long(hex_bytestr, 16)