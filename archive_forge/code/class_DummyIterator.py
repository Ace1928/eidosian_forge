import random
import hashlib
import os.path
from io import FileIO as file
from libcloud.utils.py3 import b
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class DummyIterator:

    def __init__(self, data=None):
        self.hash = hashlib.md5()
        self._data = data or []
        self._current_item = 0

    def get_md5_hash(self):
        return self.hash.hexdigest()

    def next(self):
        if self._current_item == len(self._data):
            raise StopIteration
        value = self._data[self._current_item]
        self.hash.update(b(value))
        self._current_item += 1
        return value

    def __next__(self):
        return self.next()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass