import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
class CrtCrc32Checksum(BaseChecksum):

    def __init__(self):
        self._int_crc32 = 0

    def update(self, chunk):
        new_checksum = crt_checksums.crc32(chunk, self._int_crc32)
        self._int_crc32 = new_checksum & 4294967295

    def digest(self):
        return self._int_crc32.to_bytes(4, byteorder='big')