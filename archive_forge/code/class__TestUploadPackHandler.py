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
class _TestUploadPackHandler:

    def __init__(self, backend, args, proto, stateless_rpc=None, advertise_refs=False) -> None:
        self.args = args
        self.proto = proto
        self.stateless_rpc = stateless_rpc
        self.advertise_refs = advertise_refs

    def handle(self):
        self.proto.write(b'handled input: ' + self.proto.recv(1024))