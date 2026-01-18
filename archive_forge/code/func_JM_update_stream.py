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
def JM_update_stream(doc, obj, buffer_, compress):
    """
    update a stream object
    compress stream when beneficial
    """
    len_, _ = mupdf.fz_buffer_storage(buffer_)
    nlen = len_
    if len_ > 30:
        nres = JM_compress_buffer(buffer_)
        assert isinstance(nres, mupdf.FzBuffer)
        nlen, _ = mupdf.fz_buffer_storage(nres)
    if nlen < len_ and nres and (compress == 1):
        mupdf.pdf_dict_put(obj, mupdf.PDF_ENUM_NAME_Filter, mupdf.PDF_ENUM_NAME_FlateDecode)
        mupdf.pdf_update_stream(doc, obj, nres, 1)
    else:
        mupdf.pdf_update_stream(doc, obj, buffer_, 0)