from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
@classproperty
def _hash_prefix(cls):
    """tell StaticHandler to strip ident from checksum"""
    return cls.ident