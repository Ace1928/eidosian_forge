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
def JM_object_to_buffer(what, compress, ascii):
    res = mupdf.fz_new_buffer(512)
    out = mupdf.FzOutput(res)
    mupdf.pdf_print_obj(out, what, compress, ascii)
    out.fz_close_output()
    mupdf.fz_terminate_buffer(res)
    return res