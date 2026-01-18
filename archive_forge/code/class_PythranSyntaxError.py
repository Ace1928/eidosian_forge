from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
class PythranSyntaxError(SyntaxError):

    def __init__(self, msg, node=None):
        SyntaxError.__init__(self, msg)
        if node:
            self.filename = getattr(node, 'filename', None)
            self.lineno = node.lineno
            self.offset = node.col_offset

    def __str__(self):
        loc_info = self.lineno is not None and self.offset is not None
        if self.filename and loc_info:
            with open(self.filename) as f:
                for i in range(self.lineno - 1):
                    f.readline()
                extra = '{}\n{}'.format(f.readline().rstrip(), ' ' * self.offset + '^~~~ (o_0)')
        else:
            extra = None
        if loc_info:
            format_header = '{}:{}:{}'
            format_args = (self.lineno, self.offset, self.args[0])
        else:
            format_header = '{}:'
            format_args = (self.args[0],)
        r = (format_header + ' error: {}').format(self.filename or '<unknown>', *format_args)
        if extra is not None:
            r += '\n----\n'
            r += extra
            r += '\n----\n'
        return r