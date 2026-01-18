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
def JM_append_rune(buff, ch):
    """
    APPEND non-ascii runes in unicode escape format to fz_buffer.
    """
    mupdf.fz_append_string(buff, make_escape(ch))