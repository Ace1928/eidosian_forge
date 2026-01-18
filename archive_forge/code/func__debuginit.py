import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def _debuginit(self, exc_value=None, exc_type=None, exc_tb=None, captureVars=False, Failure__init__=Failure.__init__):
    """
    Initialize failure object, possibly spawning pdb.
    """
    if (exc_value, exc_type, exc_tb) == (None, None, None):
        exc = sys.exc_info()
        if not exc[0] == self.__class__ and DO_POST_MORTEM:
            try:
                strrepr = str(exc[1])
            except BaseException:
                strrepr = 'broken str'
            print("Jumping into debugger for post-mortem of exception '{}':".format(strrepr))
            import pdb
            pdb.post_mortem(exc[2])
    Failure__init__(self, exc_value, exc_type, exc_tb, captureVars)