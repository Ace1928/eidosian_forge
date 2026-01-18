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
def _import_pkcs1_private(encoded, *kwargs):
    der = DerSequence().decode(encoded, nr_elements=9, only_ints_expected=True)
    if der[0] != 0:
        raise ValueError('No PKCS#1 encoding of an RSA private key')
    return construct(der[1:6] + [Integer(der[4]).inverse(der[5])])