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
def _forget_page(self, page: Page):
    """Remove a page from document page dict."""
    pid = id(page)
    if pid in self._page_refs:
        del self._page_refs[pid]