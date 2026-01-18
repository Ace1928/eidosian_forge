from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class NumConst(object):
    """Global info about a Python number constant held by GlobalState.

    cname       string
    value       string
    py_type     string     int, long, float
    value_code  string     evaluation code if different from value
    """

    def __init__(self, cname, value, py_type, value_code=None):
        self.cname = cname
        self.value = value
        self.py_type = py_type
        self.value_code = value_code or value