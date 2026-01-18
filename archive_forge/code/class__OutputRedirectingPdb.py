import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
class _OutputRedirectingPdb(pdb.Pdb):
    """
    A specialized version of the python debugger that redirects stdout
    to a given stream when interacting with the user.  Stdout is *not*
    redirected when traced code is executed.
    """

    def __init__(self, out):
        self.__out = out
        self.__debugger_used = False
        pdb.Pdb.__init__(self, stdout=out, nosigint=True)
        self.use_rawinput = 1

    def set_trace(self, frame=None):
        self.__debugger_used = True
        if frame is None:
            frame = sys._getframe().f_back
        pdb.Pdb.set_trace(self, frame)

    def set_continue(self):
        if self.__debugger_used:
            pdb.Pdb.set_continue(self)

    def trace_dispatch(self, *args):
        save_stdout = sys.stdout
        sys.stdout = self.__out
        try:
            return pdb.Pdb.trace_dispatch(self, *args)
        finally:
            sys.stdout = save_stdout