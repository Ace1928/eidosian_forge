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
class Drawpath(object):
    """Reflects a path dictionary from get_cdrawings()."""

    def __init__(self, **args):
        self.__dict__.update(args)