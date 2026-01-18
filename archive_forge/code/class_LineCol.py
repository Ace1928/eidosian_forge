from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class LineCol(object):
    attrib = line_col_attrib

    def __init__(self):
        self.line = None
        self.col = None
        self.data = None

    def add_kv_line_col(self, key, data):
        if self.data is None:
            self.data = {}
        self.data[key] = data

    def key(self, k):
        return self._kv(k, 0, 1)

    def value(self, k):
        return self._kv(k, 2, 3)

    def _kv(self, k, x0, x1):
        if self.data is None:
            return None
        data = self.data[k]
        return (data[x0], data[x1])

    def item(self, idx):
        if self.data is None:
            return None
        return (self.data[idx][0], self.data[idx][1])

    def add_idx_line_col(self, key, data):
        if self.data is None:
            self.data = {}
        self.data[key] = data