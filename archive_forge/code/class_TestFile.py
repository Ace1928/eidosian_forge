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
class TestFile:

    def __init__(self, exc_class) -> None:
        self.closed = False
        self._exc_class = exc_class

    def read(self, size=-1):
        raise self._exc_class

    def close(self):
        self.closed = True