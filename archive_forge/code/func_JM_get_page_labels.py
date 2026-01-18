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
def JM_get_page_labels(liste, nums):
    n = mupdf.pdf_array_len(nums)
    for i in range(0, n, 2):
        key = mupdf.pdf_resolve_indirect(mupdf.pdf_array_get(nums, i))
        pno = mupdf.pdf_to_int(key)
        val = mupdf.pdf_resolve_indirect(mupdf.pdf_array_get(nums, i + 1))
        res = JM_object_to_buffer(val, 1, 0)
        c = mupdf.fz_buffer_extract(res)
        assert isinstance(c, bytes)
        c = c.decode('utf-8')
        liste.append((pno, c))