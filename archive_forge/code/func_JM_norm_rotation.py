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
def JM_norm_rotation(rotate):
    """
    # return normalized /Rotate value:one of 0, 90, 180, 270
    """
    while rotate < 0:
        rotate += 360
    while rotate >= 360:
        rotate -= 360
    if rotate % 90 != 0:
        return 0
    return rotate