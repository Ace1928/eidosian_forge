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
def ASSERT_PDF(cond):
    assert isinstance(cond, (mupdf.PdfPage, mupdf.PdfDocument)), f'type(cond)={type(cond)!r} cond={cond!r}'
    if not cond.m_internal:
        raise Exception(MSG_IS_NO_PDF)