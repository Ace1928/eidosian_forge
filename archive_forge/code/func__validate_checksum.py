import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _validate_checksum(self):
    if self._checksum.digest() != base64.b64decode(self._expected):
        error_msg = f'Expected checksum {self._expected} did not match calculated checksum: {self._checksum.b64digest()}'
        raise FlexibleChecksumError(error_msg=error_msg)