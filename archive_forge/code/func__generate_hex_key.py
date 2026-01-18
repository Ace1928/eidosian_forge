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
def _generate_hex_key(self, key_length):
    length = int(key_length / 4)
    hex_encoded = self._generate_password(length=length, symbolgroups='0123456789ABCDEF')
    return hex_encoded