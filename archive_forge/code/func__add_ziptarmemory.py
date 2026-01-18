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
def _add_ziptarmemory(self, memory, type_, path=None):
    buff = JM_BufferFromBytes(memory)
    stream = mupdf.fz_open_buffer(buff)
    if type_ == 1:
        sub = mupdf.fz_open_zip_archive_with_stream(stream)
    else:
        sub = mupdf.fz_open_tar_archive_with_stream(stream)
    mupdf.fz_mount_multi_archive(self.this, sub, path)