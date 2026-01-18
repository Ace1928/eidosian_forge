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
def _make_page_map(self):
    """Make an array page number -> page object."""
    if self.is_closed:
        raise ValueError('document closed')
    assert 0, f'_make_page_map() is no-op'