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
def _generate_public_and_private_key(self, length, name):
    crypto_private_key = rsa.generate_private_key(public_exponent=65537, key_size=length, backend=backends.default_backend())
    private_der = crypto_private_key.private_bytes(encoding=serialization.Encoding.DER, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    crypto_public_key = crypto_private_key.public_key()
    public_der = crypto_public_key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    private_key = pri_key.PrivateKey(algorithm='RSA', bit_length=length, key=bytearray(private_der), name=name)
    public_key = pub_key.PublicKey(algorithm='RSA', bit_length=length, key=bytearray(public_der), name=name)
    return (private_key, public_key)