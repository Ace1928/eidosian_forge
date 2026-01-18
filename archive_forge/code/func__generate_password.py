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
def _generate_password(self, length, symbolgroups):
    """Generate a random password from the supplied symbol groups.

        At least one symbol from each group will be included. Unpredictable
        results if length is less than the number of symbol groups.

        Believed to be reasonably secure (with a reasonable password length!)
        """
    password = [random.choice(s) for s in symbolgroups]
    random.shuffle(password)
    password = password[:length]
    length -= len(password)
    symbols = ''.join(symbolgroups)
    password.extend([random.choice(symbols) for _i in range(length)])
    random.shuffle(password)
    return ''.join(password)