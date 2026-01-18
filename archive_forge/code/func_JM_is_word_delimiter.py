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
def JM_is_word_delimiter(ch, delimiters):
    """Check if ch is an extra word delimiting character.
    """
    if ch <= 32 or ch == 160:
        return True
    if not delimiters:
        return False
    char = chr(ch)
    for d in delimiters:
        if d == char:
            return True
    return False