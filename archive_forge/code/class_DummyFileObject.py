import random
import hashlib
import os.path
from io import FileIO as file
from libcloud.utils.py3 import b
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class DummyFileObject(file):

    def __init__(self, yield_count=5, chunk_len=10):
        self._yield_count = yield_count
        self._chunk_len = chunk_len

    def read(self, size):
        i = 0
        while i < self._yield_count:
            yield self._get_chunk(self._chunk_len)
            i += 1

    def _get_chunk(self, chunk_len):
        chunk = [str(x) for x in random.randint(97, 120)]
        return chunk

    def __len__(self):
        return self._yield_count * self._chunk_len