import binascii
import struct
import itertools
from Cryptodome.Util.py3compat import bchr, bord, tobytes, tostr, iter_range
from Cryptodome import Random
from Cryptodome.IO import PKCS8, PEM
from Cryptodome.Hash import SHA256
from Cryptodome.Util.asn1 import (
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime, COMPOSITE,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
def _import_key_der(key_data, passphrase, params):
    """Import a DSA key (public or private half), encoded in DER form."""
    decodings = (_import_openssl_private, _import_subjectPublicKeyInfo, _import_x509_cert, _import_pkcs8)
    for decoding in decodings:
        try:
            return decoding(key_data, passphrase, params)
        except ValueError:
            pass
    raise ValueError('DSA key format is not supported')