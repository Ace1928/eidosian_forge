import binascii
import copy
import random
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import private_key as pri_key
from castellan.common.objects import public_key as pub_key
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import key_manager
def _generate_key(self, **kwargs):
    name = kwargs.get('name', None)
    algorithm = kwargs.get('algorithm', 'AES')
    key_length = kwargs.get('length', 256)
    _hex = self._generate_hex_key(key_length)
    return sym_key.SymmetricKey(algorithm, key_length, bytes(binascii.unhexlify(_hex)), name)