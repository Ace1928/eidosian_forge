import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
class FileWrapper(object):

    def __init__(self, file_object):
        self.fd = file_object

    @property
    def len(self):
        return total_len(self.fd) - self.fd.tell()

    def read(self, length=-1):
        return self.fd.read(length)