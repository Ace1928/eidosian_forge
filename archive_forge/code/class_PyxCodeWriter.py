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
class PyxCodeWriter(object):
    """
    Can be used for writing out some Cython code.
    """

    def __init__(self, buffer=None, indent_level=0, context=None, encoding='ascii'):
        self.buffer = buffer or StringIOTree()
        self.level = indent_level
        self.original_level = indent_level
        self.context = context
        self.encoding = encoding

    def indent(self, levels=1):
        self.level += levels
        return True

    def dedent(self, levels=1):
        self.level -= levels

    @contextmanager
    def indenter(self, line):
        """
        with pyx_code.indenter("for i in range(10):"):
            pyx_code.putln("print i")
        """
        self.putln(line)
        self.indent()
        yield
        self.dedent()

    def empty(self):
        return self.buffer.empty()

    def getvalue(self):
        result = self.buffer.getvalue()
        if isinstance(result, bytes):
            result = result.decode(self.encoding)
        return result

    def putln(self, line, context=None):
        context = context or self.context
        if context:
            line = sub_tempita(line, context)
        self._putln(line)

    def _putln(self, line):
        self.buffer.write(u'%s%s\n' % (self.level * u'    ', line))

    def put_chunk(self, chunk, context=None):
        context = context or self.context
        if context:
            chunk = sub_tempita(chunk, context)
        chunk = textwrap.dedent(chunk)
        for line in chunk.splitlines():
            self._putln(line)

    def insertion_point(self):
        return type(self)(self.buffer.insertion_point(), self.level, self.context)

    def reset(self):
        self.buffer.reset()
        self.level = self.original_level

    def named_insertion_point(self, name):
        setattr(self, name, self.insertion_point())