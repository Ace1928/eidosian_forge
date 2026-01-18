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
class IncludeCode(object):
    """
    An include file and/or verbatim C code to be included in the
    generated sources.
    """
    INITIAL = 0
    EARLY = 1
    LATE = 2
    counter = 1

    def __init__(self, include=None, verbatim=None, late=True, initial=False):
        self.order = self.counter
        type(self).counter += 1
        self.pieces = {}
        if include:
            if include[0] == '<' and include[-1] == '>':
                self.pieces[0] = u'#include {0}'.format(include)
                late = False
            else:
                self.pieces[0] = u'#include "{0}"'.format(include)
        if verbatim:
            self.pieces[self.order] = verbatim
        if initial:
            self.location = self.INITIAL
        elif late:
            self.location = self.LATE
        else:
            self.location = self.EARLY

    def dict_update(self, d, key):
        """
        Insert `self` in dict `d` with key `key`. If that key already
        exists, update the attributes of the existing value with `self`.
        """
        if key in d:
            other = d[key]
            other.location = min(self.location, other.location)
            other.pieces.update(self.pieces)
        else:
            d[key] = self

    def sortkey(self):
        return self.order

    def mainpiece(self):
        """
        Return the main piece of C code, corresponding to the include
        file. If there was no include file, return None.
        """
        return self.pieces.get(0)

    def write(self, code):
        for k in sorted(self.pieces):
            code.putln(self.pieces[k])