from hashlib import sha256, sha384, sha512
from paramiko.common import byte_chr
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from binascii import hexlify
def _generate_key_pair(self):
    self.P = ec.generate_private_key(self.curve, default_backend())
    if self.transport.server_mode:
        self.Q_S = self.P.public_key()
        return
    self.Q_C = self.P.public_key()