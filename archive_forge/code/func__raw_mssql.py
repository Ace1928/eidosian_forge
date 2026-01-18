from binascii import hexlify, unhexlify
from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import consteq
from passlib.utils.compat import bascii_to_str, unicode, u
import passlib.utils.handlers as uh
def _raw_mssql(secret, salt):
    assert isinstance(secret, unicode)
    assert isinstance(salt, bytes)
    return sha1(secret.encode('utf-16-le') + salt).digest()