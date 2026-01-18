from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import hashlib
import os
import six
from boto import config
import crcmod
from gslib.exception import CommandException
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
def GetMd5(byte_string=b''):
    """Returns md5 object, avoiding incorrect FIPS error on Red Hat systems.

  Examples: GetMd5(b'abc')
            GetMd5(bytes('abc', encoding='utf-8'))

  Args:
    byte_string (bytes): String in bytes form to hash. Don't include for empty
      hash object, since md5(b'').digest() == md5().digest().

  Returns:
    md5 hash object.
  """
    try:
        return hashlib.md5(byte_string)
    except ValueError:
        return hashlib.md5(byte_string, usedforsecurity=False)