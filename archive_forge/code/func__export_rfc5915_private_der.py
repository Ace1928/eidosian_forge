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
def _export_rfc5915_private_der(self, include_ec_params=True):
    assert self.has_private()
    modulus_bytes = self.pointQ.size_in_bytes()
    public_key = b'\x04' + self.pointQ.x.to_bytes(modulus_bytes) + self.pointQ.y.to_bytes(modulus_bytes)
    seq = [1, DerOctetString(self.d.to_bytes(modulus_bytes)), DerObjectId(self._curve.oid, explicit=0), DerBitString(public_key, explicit=1)]
    if not include_ec_params:
        del seq[2]
    return DerSequence(seq).encode()