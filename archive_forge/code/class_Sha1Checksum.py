import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
class Sha1Checksum(BaseChecksum):

    def __init__(self):
        self._checksum = sha1()

    def update(self, chunk):
        self._checksum.update(chunk)

    def digest(self):
        return self._checksum.digest()