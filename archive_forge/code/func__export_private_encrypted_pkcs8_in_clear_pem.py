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
def _export_private_encrypted_pkcs8_in_clear_pem(self, passphrase, **kwargs):
    from Cryptodome.IO import PEM
    assert passphrase
    if 'protection' not in kwargs:
        raise ValueError("At least the 'protection' parameter should be present")
    encoded_der = self._export_pkcs8(passphrase=passphrase, **kwargs)
    return PEM.encode(encoded_der, 'ENCRYPTED PRIVATE KEY')