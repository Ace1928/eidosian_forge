from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
class Nprint(object):

    def __init__(self, file_name=None):
        self._max_print = None
        self._count = None
        self._file_name = file_name

    def __call__(self, *args, **kw):
        if not bool(_debug):
            return
        out = sys.stdout if self._file_name is None else open(self._file_name, 'a')
        dbgprint = print
        kw1 = kw.copy()
        kw1['file'] = out
        dbgprint(*args, **kw1)
        out.flush()
        if self._max_print is not None:
            if self._count is None:
                self._count = self._max_print
            self._count -= 1
            if self._count == 0:
                dbgprint('forced exit\n')
                traceback.print_stack()
                out.flush()
                sys.exit(0)
        if self._file_name:
            out.close()

    def set_max_print(self, i):
        self._max_print = i
        self._count = None