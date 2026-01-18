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
def CheckFontInfo(doc: Document, xref: int) -> list:
    """Return a font info if present in the document.
    """
    for f in doc.FontInfos:
        if xref == f[0]:
            return f