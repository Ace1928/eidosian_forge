import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class _MsgPart(object):

    def __init__(self, line, tok):
        assert line >= 0
        self.line = line
        self.tok = tok

    @classmethod
    def add_to_line_to_contents(cls, obj, line_to_contents, line=None):
        if isinstance(obj, (list, tuple)):
            for o in obj:
                cls.add_to_line_to_contents(o, line_to_contents, line=line)
            return
        if isinstance(obj, str):
            assert line is not None
            line = int(line)
            lst = line_to_contents.setdefault(line, [])
            lst.append(obj)
            return
        if isinstance(obj, _MsgPart):
            if isinstance(obj.tok, (list, tuple)):
                cls.add_to_line_to_contents(obj.tok, line_to_contents, line=obj.line)
                return
            if isinstance(obj.tok, str):
                lst = line_to_contents.setdefault(obj.line, [])
                lst.append(obj.tok)
                return
        raise AssertionError('Unhandled: %' % (obj,))