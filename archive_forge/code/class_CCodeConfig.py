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
class CCodeConfig(object):

    def __init__(self, emit_linenums=True, emit_code_comments=True, c_line_in_traceback=True):
        self.emit_code_comments = emit_code_comments
        self.emit_linenums = emit_linenums
        self.c_line_in_traceback = c_line_in_traceback