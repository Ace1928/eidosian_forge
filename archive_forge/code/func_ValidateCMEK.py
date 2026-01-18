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
def ValidateCMEK(key):
    if not key:
        raise CommandException('KMS key is empty.')
    if key.startswith('/'):
        raise CommandException('KMS key should not start with leading slash (/): "%s"' % key)
    if not VALID_CMEK_RE().match(key):
        raise CommandException('Invalid KMS key name: "%s".\nKMS keys should follow the format "projects/<project-id>/locations/<location>/keyRings/<keyring>/cryptoKeys/<key-name>"' % key)