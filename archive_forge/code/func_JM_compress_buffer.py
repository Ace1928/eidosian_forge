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
def JM_compress_buffer(inbuffer):
    """
    compress char* into a new buffer
    """
    data, compressed_length = mupdf.fz_new_deflated_data_from_buffer(inbuffer, mupdf.FZ_DEFLATE_BEST)
    if not data or compressed_length == 0:
        return None
    buf = mupdf.FzBuffer(mupdf.fz_new_buffer_from_data(data, compressed_length))
    mupdf.fz_resize_buffer(buf, compressed_length)
    return buf