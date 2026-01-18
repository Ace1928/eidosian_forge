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
def _loadOutline(self):
    """Load first outline."""
    doc = self.this
    assert isinstance(doc, mupdf.FzDocument)
    try:
        ol = mupdf.fz_load_outline(doc)
    except Exception:
        if g_exceptions_verbose > 1:
            exception_info()
        return
    return Outline(ol)