import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
class _ScriptTarget(str):

    def __new__(cls, val):
        res = super().__new__(cls, os.path.realpath(val))
        res.orig = val
        return res

    def check(self):
        if not os.path.exists(self):
            print('Error:', self.orig, 'does not exist')
            sys.exit(1)
        if os.path.isdir(self):
            print('Error:', self.orig, 'is a directory')
            sys.exit(1)
        sys.path[0] = os.path.dirname(self)

    @property
    def filename(self):
        return self

    @property
    def namespace(self):
        return dict(__name__='__main__', __file__=self, __builtins__=__builtins__)

    @property
    def code(self):
        with io.open_code(self) as fp:
            return f"exec(compile({fp.read()!r}, {self!r}, 'exec'))"