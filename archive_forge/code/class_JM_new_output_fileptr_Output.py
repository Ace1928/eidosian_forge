import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class JM_new_output_fileptr_Output(mupdf.FzOutput2):

    def __init__(self, bio):
        super().__init__()
        self.bio = bio
        self.use_virtual_write()
        self.use_virtual_seek()
        self.use_virtual_tell()
        self.use_virtual_truncate()

    def seek(self, ctx, offset, whence):
        return self.bio.seek(offset, whence)

    def tell(self, ctx):
        ret = self.bio.tell()
        return ret

    def truncate(self, ctx):
        return self.bio.truncate()

    def write(self, ctx, data_raw, data_length):
        data = mupdf.raw_to_python_bytes(data_raw, data_length)
        return self.bio.write(data)