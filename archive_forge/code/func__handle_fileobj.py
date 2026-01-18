import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _handle_fileobj(self, fileobj):
    start_position = fileobj.tell()
    for chunk in iter(lambda: fileobj.read(self._CHUNK_SIZE), b''):
        self.update(chunk)
    fileobj.seek(start_position)