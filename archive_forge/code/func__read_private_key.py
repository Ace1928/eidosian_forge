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
def _read_private_key(self, tag, f, password=None):
    lines = f.readlines()
    if not lines:
        raise SSHException('no lines in {} private key file'.format(tag))
    start = 0
    m = self.BEGIN_TAG.match(lines[start])
    line_range = len(lines) - 1
    while start < line_range and (not m):
        start += 1
        m = self.BEGIN_TAG.match(lines[start])
    start += 1
    keytype = m.group(1) if m else None
    if start >= len(lines) or keytype is None:
        raise SSHException('not a valid {} private key file'.format(tag))
    end = start
    m = self.END_TAG.match(lines[end])
    while end < line_range and (not m):
        end += 1
        m = self.END_TAG.match(lines[end])
    if keytype == tag:
        data = self._read_private_key_pem(lines, end, password)
        pkformat = self._PRIVATE_KEY_FORMAT_ORIGINAL
    elif keytype == 'OPENSSH':
        data = self._read_private_key_openssh(lines[start:end], password)
        pkformat = self._PRIVATE_KEY_FORMAT_OPENSSH
    else:
        raise SSHException('encountered {} key, expected {} key'.format(keytype, tag))
    return (pkformat, data)