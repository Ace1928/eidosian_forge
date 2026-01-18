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
@property
def abs_unit(self):
    """Unit vector with positive coordinates."""
    s = self.x * self.x + self.y * self.y
    if s < EPSILON:
        return Point(0, 0)
    s = math.sqrt(s)
    return Point(abs(self.x) / s, abs(self.y) / s)