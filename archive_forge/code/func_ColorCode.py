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
def ColorCode(c: typing.Union[list, tuple, float, None], f: str) -> str:
    if not c:
        return ''
    if hasattr(c, '__float__'):
        c = (c,)
    CheckColor(c)
    if len(c) == 1:
        s = '%g ' % c[0]
        return s + 'G ' if f == 'c' else s + 'g '
    if len(c) == 3:
        s = '%g %g %g ' % tuple(c)
        return s + 'RG ' if f == 'c' else s + 'rg '
    s = '%g %g %g %g ' % tuple(c)
    return s + 'K ' if f == 'c' else s + 'k '