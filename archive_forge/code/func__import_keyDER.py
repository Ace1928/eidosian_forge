import binascii
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import tobytes, bord, tostr
from Cryptodome.Util.asn1 import DerSequence, DerNull
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
def _import_keyDER(extern_key, passphrase):
    """Import an RSA key (public or private half), encoded in DER form."""
    decodings = (_import_pkcs1_private, _import_pkcs1_public, _import_subjectPublicKeyInfo, _import_x509_cert, _import_pkcs8)
    for decoding in decodings:
        try:
            return decoding(extern_key, passphrase)
        except ValueError:
            pass
    raise ValueError('RSA key format is not supported')