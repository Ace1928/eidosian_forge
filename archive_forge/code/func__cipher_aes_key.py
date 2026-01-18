from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
@staticmethod
def _cipher_aes_key(value, secret, salt, cost, decrypt=False):
    """
        Internal helper for :meth:`encrypt_key` --
        handles lowlevel encryption/decryption.

        Algorithm details:

        This function uses PBKDF2-HMAC-SHA256 to generate a 32-byte AES key
        and a 16-byte IV from the application secret & random salt.
        It then uses AES-256-CTR to encrypt/decrypt the TOTP key.

        CTR mode was chosen over CBC because the main attack scenario here
        is that the attacker has stolen the database, and is trying to decrypt a TOTP key
        (the plaintext value here).  To make it hard for them, we want every password
        to decrypt to a potentially valid key -- thus need to avoid any authentication
        or padding oracle attacks.  While some random padding construction could be devised
        to make this work for CBC mode, a stream cipher mode is just plain simpler.
        OFB/CFB modes would also work here, but seeing as they have malleability
        and cyclic issues (though remote and barely relevant here),
        CTR was picked as the best overall choice.
        """
    if _cg_ciphers is None:
        raise RuntimeError("TOTP encryption requires 'cryptography' package (https://cryptography.io)")
    keyiv = pbkdf2_hmac('sha256', secret, salt=salt, rounds=1 << cost, keylen=48)
    cipher = _cg_ciphers.Cipher(_cg_ciphers.algorithms.AES(keyiv[:32]), _cg_ciphers.modes.CTR(keyiv[32:]), _cg_default_backend())
    ctx = cipher.decryptor() if decrypt else cipher.encryptor()
    return ctx.update(value) + ctx.finalize()