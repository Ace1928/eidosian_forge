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
def JM_font_name(font):
    assert isinstance(font, mupdf.FzFont)
    name = mupdf.fz_font_name(font)
    s = name.find('+')
    if g_subset_fontnames or s == -1 or s != 6:
        return name
    return name[s + 1:]