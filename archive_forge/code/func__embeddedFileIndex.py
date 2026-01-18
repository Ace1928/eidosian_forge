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
def _embeddedFileIndex(self, item: typing.Union[int, str]) -> int:
    filenames = self.embfile_names()
    msg = "'%s' not in EmbeddedFiles array." % str(item)
    if item in filenames:
        idx = filenames.index(item)
    elif item in range(len(filenames)):
        idx = item
    else:
        raise ValueError(msg)
    return idx