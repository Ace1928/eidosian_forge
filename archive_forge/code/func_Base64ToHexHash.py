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
def Base64ToHexHash(base64_hash):
    """Returns the hex digest value of the input base64-encoded hash.

  Args:
    base64_hash: Base64-encoded hash, which may contain newlines and single or
        double quotes.

  Returns:
    Hex digest of the input argument.
  """
    decoded_bytes = base64.b64decode(base64_hash.strip('\n"\'').encode(UTF8))
    return binascii.hexlify(decoded_bytes)