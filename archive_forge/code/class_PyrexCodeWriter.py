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
class PyrexCodeWriter(object):

    def __init__(self, outfile_name):
        self.f = Utils.open_new_file(outfile_name)
        self.level = 0

    def putln(self, code):
        self.f.write('%s%s\n' % (' ' * self.level, code))

    def indent(self):
        self.level += 1

    def dedent(self):
        self.level -= 1