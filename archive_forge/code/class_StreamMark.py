from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
class StreamMark(object):
    __slots__ = ('name', 'index', 'line', 'column')

    def __init__(self, name, index, line, column):
        self.name = name
        self.index = index
        self.line = line
        self.column = column

    def __str__(self):
        where = '  in "%s", line %d, column %d' % (self.name, self.line + 1, self.column + 1)
        return where