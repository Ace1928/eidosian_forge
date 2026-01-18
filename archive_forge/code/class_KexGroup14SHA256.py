from paramiko.kex_group1 import KexGroup1
from hashlib import sha1, sha256
class KexGroup14SHA256(KexGroup14):
    name = 'diffie-hellman-group14-sha256'
    hash_algo = sha256