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
def JM_append_word(lines, buff, wbbox, block_n, line_n, word_n):
    """
    Functions for wordlist output
    """
    s = JM_EscapeStrFromBuffer(buff)
    litem = (wbbox.x0, wbbox.y0, wbbox.x1, wbbox.y1, s, block_n, line_n, word_n)
    lines.append(litem)
    return (word_n + 1, mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY))