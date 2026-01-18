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
def INFINITE_IRECT():
    return IRect(FZ_MIN_INF_RECT, FZ_MIN_INF_RECT, FZ_MAX_INF_RECT, FZ_MAX_INF_RECT)