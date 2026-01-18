from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
class CursorFile(object):

    def __init__(self, config):
        self._fobj = io.StringIO()
        self._cursor = Cursor(0, 0)
        self._config = config

    @property
    def cursor(self):
        return Cursor(*self._cursor)

    def assert_at(self, cursor):
        assert self._cursor[0] == cursor[0] and self._cursor[1] == cursor[1], 'self._cursor=({},{}), write_at=({}, {}):\n{}'.format(self._cursor[0], self._cursor[1], cursor[0], cursor[1], self.getvalue())

    def assert_lt(self, cursor):
        assert self._cursor[0] < cursor[0] or (self._cursor[0] == cursor[0] and self._cursor[1] <= cursor[1]), 'self._cursor=({},{}), write_at=({}, {}):\n{}'.format(self._cursor[0], self._cursor[1], cursor[0], cursor[1], self.getvalue().encode('utf-8', errors='replace'))

    def forge_cursor(self, cursor):
        self._cursor = cursor

    def write_at(self, cursor, text):
        if sys.version_info[0] < 3 and isinstance(text, str):
            text = text.decode('utf-8')
        self.assert_lt(cursor)
        rows = cursor[0] - self._cursor[0]
        if rows:
            self._fobj.write(self._config.format.endl * rows)
            self._cursor[0] += rows
            self._cursor[1] = 0
        cols = cursor[1] - self._cursor[1]
        if cols:
            self._fobj.write(' ' * cols)
            self._cursor[1] += cols
        lines = text.split('\n')
        line = lines.pop(0)
        self._fobj.write(line)
        self._cursor[1] += len(line)
        while lines:
            self._fobj.write(self._config.format.endl)
            self._cursor[0] += 1
            self._cursor[1] = 0
            line = lines.pop(0)
            self._fobj.write(line)
            self._cursor[1] += len(line)

    def write(self, copy_text):
        if sys.version_info[0] < 3 and isinstance(copy_text, str):
            copy_text = copy_text.decode('utf-8')
        self._fobj.write(copy_text)
        if '\n' not in copy_text:
            self._cursor[1] += len(copy_text)
        else:
            self._cursor[0] += copy_text.count('\n')
            self._cursor[1] = len(copy_text.split('\n')[-1])

    def getvalue(self):
        return self._fobj.getvalue() + self._config.format.endl