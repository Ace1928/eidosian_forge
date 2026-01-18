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
def NormalizeStorageClass(sc):
    """Returns a normalized form of the given storage class name.

  Converts the given string to uppercase and expands valid abbreviations to
  full storage class names (e.g 'std' would return 'STANDARD'). Note that this
  method does not check if the given storage class is valid.

  Args:
    sc: (str) String representing the storage class's full name or abbreviation.

  Returns:
    (str) A string representing the full name of the given storage class.
  """
    sc = sc.upper()
    if sc in STORAGE_CLASS_SHORTHAND_TO_FULL_NAME:
        sc = STORAGE_CLASS_SHORTHAND_TO_FULL_NAME[sc]
    return sc