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
def InsistAscii(string, message):
    """Ensures that the string passed in consists of only ASCII values.

  Args:
    string: Union[str, unicode, bytes] Text that will be checked for
        ASCII values.
    message: Union[str, unicode, bytes] Error message, passed into the
        exception, in the event that the check on `string` fails.

  Returns:
    None

  Raises:
    CommandException
  """
    if not all((ord(c) < 128 for c in string)):
        raise CommandException(message)