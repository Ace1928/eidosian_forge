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
def _import_subjectPublicKeyInfo(encoded, *kwargs):
    """Convert a subjectPublicKeyInfo into an EccKey object"""
    oid, ec_point, params = _expand_subject_public_key_info(encoded)
    nist_p_oids = ('1.2.840.10045.2.1', '1.3.132.1.12', '1.3.132.1.13')
    eddsa_oids = {'1.3.101.112': ('Ed25519', _import_ed25519_public_key), '1.3.101.113': ('Ed448', _import_ed448_public_key)}
    if oid in nist_p_oids:
        if not params:
            raise ValueError('Missing ECC parameters for ECC OID %s' % oid)
        try:
            curve_oid = DerObjectId().decode(params).value
        except ValueError:
            raise ValueError('Error decoding namedCurve')
        return _import_public_der(ec_point, curve_oid=curve_oid)
    elif oid in eddsa_oids:
        curve_name, import_eddsa_public_key = eddsa_oids[oid]
        if params:
            raise ValueError('Unexpected ECC parameters for ECC OID %s' % oid)
        x, y = import_eddsa_public_key(ec_point)
        return construct(point_x=x, point_y=y, curve=curve_name)
    else:
        raise UnsupportedEccFeature('Unsupported ECC OID: %s' % oid)