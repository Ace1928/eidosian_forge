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
def CalculateHashesFromContents(fp, hash_dict, callback_processor=None):
    """Calculates hashes of the contents of a file.

  Args:
    fp: An already-open file object (stream will be consumed).
    hash_dict: Dict of (string alg_name: initialized hashing class)
        Hashing class will be populated with digests upon return.
    callback_processor: Optional callback processing class that implements
        Progress(integer amount of bytes processed).
  """
    while True:
        data = fp.read(DEFAULT_FILE_BUFFER_SIZE)
        if not data:
            break
        if six.PY3:
            if isinstance(data, str):
                data = data.encode(UTF8)
        for hash_alg in six.itervalues(hash_dict):
            hash_alg.update(data)
        if callback_processor:
            callback_processor.Progress(len(data))