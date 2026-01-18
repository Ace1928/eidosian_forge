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
def _reset_page_refs(self):
    """Invalidate all pages in document dictionary."""
    if getattr(self, 'is_closed', True):
        return
    pages = [p for p in self._page_refs.values()]
    for page in pages:
        if page:
            page._erase()
            page = None
    self._page_refs.clear()