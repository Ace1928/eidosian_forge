from base64 import b64encode
from binascii import hexlify
from hashlib import md5, sha1, sha256
import logging; log = logging.getLogger(__name__)
from passlib.handlers.bcrypt import _wrapped_bcrypt
from passlib.hash import argon2, bcrypt, pbkdf2_sha1, pbkdf2_sha256
from passlib.utils import to_unicode, rng, getrandstr
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import str_to_uascii, uascii_to_str, unicode, u
from passlib.crypto.digest import pbkdf2_hmac
import passlib.utils.handlers as uh
def _import_des_crypt():
    global des_crypt
    if des_crypt is None:
        from passlib.hash import des_crypt
    return des_crypt