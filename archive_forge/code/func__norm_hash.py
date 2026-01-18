from binascii import hexlify, unhexlify
from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode, xor_bytes
from passlib.utils.compat import irange, u, \
from passlib.crypto.des import des_encrypt_block
import passlib.utils.handlers as uh
@classmethod
def _norm_hash(cls, hash):
    return hash.upper()