import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
class MinimalistWSGIInputStream:
    """WSGI input stream with no 'seek()' and 'tell()' methods."""

    def __init__(self, data) -> None:
        self.data = data
        self.pos = 0

    def read(self, howmuch):
        start = self.pos
        end = self.pos + howmuch
        if start >= len(self.data):
            return b''
        self.pos = end
        return self.data[start:end]