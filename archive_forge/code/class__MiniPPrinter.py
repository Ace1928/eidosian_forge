import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
class _MiniPPrinter(object):

    def __init__(self):
        self._out = StringIO()
        self.indentation = 0

    def text(self, text):
        self._out.write(text)

    def breakable(self, sep=' '):
        self._out.write(sep)

    def begin_group(self, _, text):
        self.text(text)

    def end_group(self, _, text):
        self.text(text)

    def pretty(self, obj):
        if hasattr(obj, '_repr_pretty_'):
            obj._repr_pretty_(self, False)
        else:
            self.text(repr(obj))

    def getvalue(self):
        return self._out.getvalue()