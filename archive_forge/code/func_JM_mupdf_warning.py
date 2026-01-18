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
def JM_mupdf_warning(message):
    """
    redirect MuPDF warnings
    """
    sys.stderr.flush()
    JM_mupdf_warnings_store.append(message)
    if JM_mupdf_show_warnings:
        sys.stderr.write(f'MuPDF warning: {message}\n')