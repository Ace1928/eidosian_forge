import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
class KexGexSHA256(KexGex):
    name = 'diffie-hellman-group-exchange-sha256'
    hash_algo = sha256