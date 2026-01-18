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
def CheckColor(c: OptSeq):
    if c:
        if type(c) not in (list, tuple) or len(c) not in (1, 3, 4) or min(c) < 0 or (max(c) > 1):
            raise ValueError('need 1, 3 or 4 color components in range 0 to 1')