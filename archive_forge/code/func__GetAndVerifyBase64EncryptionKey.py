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
def _GetAndVerifyBase64EncryptionKey(boto_config):
    """Reads the encryption key from boto_config and ensures it is base64-encoded.

  Args:
    boto_config: (boto.pyami.config.Config) The boto config in which to check
        for a matching encryption key.

  Returns:
    (str) Base64-encoded encryption key string, or None if no encryption key
    exists in configuration.

  """
    encryption_key = boto_config.get('GSUtil', 'encryption_key', None)
    if encryption_key:
        try:
            base64.b64decode(encryption_key)
        except:
            raise CommandException('Configured encryption_key is not a valid base64 string. Please double-check your configuration and ensure the key is valid and in base64 format.')
    return encryption_key