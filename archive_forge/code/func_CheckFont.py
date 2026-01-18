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
def CheckFont(page: Page, fontname: str) -> tuple:
    """Return an entry in the page's font list if reference name matches.
    """
    for f in page.get_fonts():
        if f[4] == fontname:
            return f