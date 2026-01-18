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
def _import_rfc5915_der(encoded, passphrase, curve_oid=None):
    private_key = DerSequence().decode(encoded, nr_elements=(3, 4))
    if private_key[0] != 1:
        raise ValueError('Incorrect ECC private key version')
    try:
        parameters = DerObjectId(explicit=0).decode(private_key[2]).value
        if curve_oid is not None and parameters != curve_oid:
            raise ValueError('Curve mismatch')
        curve_oid = parameters
    except ValueError:
        pass
    if curve_oid is None:
        raise ValueError('No curve found')
    for curve_name, curve in _curves.items():
        if curve.oid == curve_oid:
            break
    else:
        raise UnsupportedEccFeature('Unsupported ECC curve (OID: %s)' % curve_oid)
    scalar_bytes = DerOctetString().decode(private_key[1]).payload
    modulus_bytes = curve.p.size_in_bytes()
    if len(scalar_bytes) != modulus_bytes:
        raise ValueError('Private key is too small')
    d = Integer.from_bytes(scalar_bytes)
    if len(private_key) > 2:
        public_key_enc = DerBitString(explicit=1).decode(private_key[-1]).value
        public_key = _import_public_der(public_key_enc, curve_oid=curve_oid)
        point_x = public_key.pointQ.x
        point_y = public_key.pointQ.y
    else:
        point_x = point_y = None
    return construct(curve=curve_name, d=d, point_x=point_x, point_y=point_y)