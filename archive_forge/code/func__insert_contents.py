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
def _insert_contents(page, newcont, overlay=1):
    """Add bytes as a new /Contents object for a page, and return its xref."""
    pdfpage = page._pdf_page()
    ASSERT_PDF(pdfpage)
    contbuf = JM_BufferFromBytes(newcont)
    xref = JM_insert_contents(pdfpage.doc(), pdfpage.obj(), contbuf, overlay)
    return xref