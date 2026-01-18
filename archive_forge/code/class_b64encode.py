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
class b64encode(json.JSONEncoder):

    def default(self, s):
        if type(s) in (bytes, bytearray):
            return base64.b64encode(s).decode()