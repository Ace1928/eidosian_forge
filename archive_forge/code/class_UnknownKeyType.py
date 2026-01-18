import base64
from base64 import encodebytes, decodebytes
from binascii import unhexlify
import os
from pathlib import Path
from hashlib import md5, sha256
import re
import struct
import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import algorithms, modes, Cipher
from cryptography.hazmat.primitives import asymmetric
from paramiko import util
from paramiko.util import u, b
from paramiko.common import o600
from paramiko.ssh_exception import SSHException, PasswordRequiredException
from paramiko.message import Message
class UnknownKeyType(Exception):
    """
    An unknown public/private key algorithm was attempted to be read.
    """

    def __init__(self, key_type=None, key_bytes=None):
        self.key_type = key_type
        self.key_bytes = key_bytes

    def __str__(self):
        return f'UnknownKeyType(type={self.key_type!r}, bytes=<{len(self.key_bytes)}>)'