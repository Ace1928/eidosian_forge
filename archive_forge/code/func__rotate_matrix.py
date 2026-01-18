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
@staticmethod
def _rotate_matrix(page):
    pdfpage = page._pdf_page()
    if not pdfpage.m_internal:
        return JM_py_from_matrix(mupdf.FzMatrix())
    return JM_py_from_matrix(JM_rotate_page_matrix(pdfpage))