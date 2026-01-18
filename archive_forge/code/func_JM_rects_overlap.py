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
def JM_rects_overlap(a, b):
    if 0 or a.x0 >= b.x1 or a.y0 >= b.y1 or (a.x1 <= b.x0) or (a.y1 <= b.y0):
        return 0
    return 1