from hashlib import sha256, sha384, sha512
from paramiko.common import byte_chr
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from binascii import hexlify
class KexNistp521(KexNistp256):
    name = 'ecdh-sha2-nistp521'
    hash_algo = sha512
    curve = ec.SECP521R1()