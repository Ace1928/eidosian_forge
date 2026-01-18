import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
class LengthWrapper:
    """
    Wrap a filelike object with a maximum length.

    Fix for https://github.com/kennethreitz/requests/issues/1648.
    It is recommended to use this class only on files opened in binary mode.
    """

    def __init__(self, readable, length, md5=False):
        """
        :param readable: The filelike object to read from.
        :param length: The maximum amount of content that can be read from
                       the filelike object before it is simulated to be
                       empty.
        :param md5: Flag to enable calculating the MD5 of the content
                    as it is read.
        """
        self._md5 = md5
        self._reset_md5()
        self._length = self._remaining = length
        self._readable = readable
        self._can_reset = all((hasattr(readable, attr) for attr in ('seek', 'tell')))
        if self._can_reset:
            self._start = readable.tell()

    def __len__(self):
        return self._length

    def _reset_md5(self):
        self.md5sum = hashlib.md5() if self._md5 else NoopMD5()

    def get_md5sum(self):
        return self.md5sum.hexdigest()

    def read(self, size=-1):
        if self._remaining <= 0:
            return ''
        to_read = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._readable.read(to_read)
        self._remaining -= len(chunk)
        try:
            self.md5sum.update(chunk)
        except TypeError:
            self.md5sum.update(chunk.encode())
        return chunk

    @property
    def reset(self):
        if self._can_reset:
            return self._reset
        raise AttributeError("%r object has no attribute 'reset'" % type(self).__name__)

    def _reset(self, *args, **kwargs):
        if not self._can_reset:
            raise TypeError('%r object cannot be reset; needs both seek and tell methods' % type(self._readable).__name__)
        self._readable.seek(self._start)
        self._reset_md5()
        self._remaining = self._length