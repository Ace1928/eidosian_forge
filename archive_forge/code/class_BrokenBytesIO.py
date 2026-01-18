import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
class BrokenBytesIO(io.BytesIO):
    allow_write = False

    def write(self, b):
        if self.allow_write:
            return super().write(b)
        else:
            raise Exception('I am broken')