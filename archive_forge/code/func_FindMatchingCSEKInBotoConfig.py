from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
from hashlib import sha256
import re
import sys
import six
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
def FindMatchingCSEKInBotoConfig(key_sha256, boto_config):
    """Searches boto_config for a CSEK with the given base64-encoded SHA256 hash.

  Args:
    key_sha256: (str) Base64-encoded SHA256 hash of the AES256 encryption key.
    boto_config: (boto.pyami.config.Config) The boto config in which to check
        for a matching encryption key.

  Returns:
    (str) Base64-encoded encryption key string if a match is found, None
    otherwise.
  """
    if six.PY3:
        if not isinstance(key_sha256, bytes):
            key_sha256 = key_sha256.encode('ascii')
    keywrapper = CryptoKeyWrapperFromKey(boto_config.get('GSUtil', 'encryption_key', None))
    if keywrapper is not None and keywrapper.crypto_type == CryptoKeyType.CSEK and (keywrapper.crypto_key_sha256 == key_sha256):
        return keywrapper.crypto_key
    for i in range(MAX_DECRYPTION_KEYS):
        key_number = i + 1
        keywrapper = CryptoKeyWrapperFromKey(boto_config.get('GSUtil', 'decryption_key%s' % str(key_number), None))
        if keywrapper is None:
            break
        elif keywrapper.crypto_type == CryptoKeyType.CSEK and keywrapper.crypto_key_sha256 == key_sha256:
            return keywrapper.crypto_key