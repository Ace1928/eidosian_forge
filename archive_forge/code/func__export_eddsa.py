from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
def _export_eddsa(self):
    x, y = self.pointQ.xy
    if self._curve.name == 'ed25519':
        result = bytearray(y.to_bytes(32, byteorder='little'))
        result[31] = (x & 1) << 7 | result[31]
    elif self._curve.name == 'ed448':
        result = bytearray(y.to_bytes(57, byteorder='little'))
        result[56] = (x & 1) << 7
    else:
        raise ValueError('Not an EdDSA key to export')
    return bytes(result)