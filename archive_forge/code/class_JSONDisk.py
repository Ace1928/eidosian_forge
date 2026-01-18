import codecs
import contextlib as cl
import errno
import functools as ft
import io
import json
import os
import os.path as op
import pickle
import pickletools
import sqlite3
import struct
import tempfile
import threading
import time
import warnings
import zlib
class JSONDisk(Disk):
    """Cache key and value using JSON serialization with zlib compression."""

    def __init__(self, directory, compress_level=1, **kwargs):
        """Initialize JSON disk instance.

        Keys and values are compressed using the zlib library. The
        `compress_level` is an integer from 0 to 9 controlling the level of
        compression; 1 is fastest and produces the least compression, 9 is
        slowest and produces the most compression, and 0 is no compression.

        :param str directory: directory path
        :param int compress_level: zlib compression level (default 1)
        :param kwargs: super class arguments

        """
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        json_bytes = json.dumps(key).encode('utf-8')
        data = zlib.compress(json_bytes, self.compress_level)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return json.loads(zlib.decompress(data).decode('utf-8'))

    def store(self, value, read, key=UNKNOWN):
        if not read:
            json_bytes = json.dumps(value).encode('utf-8')
            value = zlib.compress(json_bytes, self.compress_level)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = json.loads(zlib.decompress(data).decode('utf-8'))
        return data