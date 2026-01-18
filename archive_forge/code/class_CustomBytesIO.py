import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
class CustomBytesIO(io.BytesIO):

    def __init__(self, buffer=None, encoding='utf-8'):
        buffer = encode_with(buffer, encoding)
        super(CustomBytesIO, self).__init__(buffer)

    def _get_end(self):
        current_pos = self.tell()
        self.seek(0, 2)
        length = self.tell()
        self.seek(current_pos, 0)
        return length

    @property
    def len(self):
        length = self._get_end()
        return length - self.tell()

    def append(self, bytes):
        with reset(self):
            written = self.write(bytes)
        return written

    def smart_truncate(self):
        to_be_read = total_len(self)
        already_read = self._get_end() - to_be_read
        if already_read >= to_be_read:
            old_bytes = self.read()
            self.seek(0, 0)
            self.truncate()
            self.write(old_bytes)
            self.seek(0, 0)