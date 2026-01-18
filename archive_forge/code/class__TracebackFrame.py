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
class _TracebackFrame:
    """
    Fake traceback object which can be passed to functions in the standard
    library L{traceback} module.
    """

    def __init__(self, frame):
        """
        @param frame: _Frame object
        """
        self.tb_frame = frame
        self.tb_lineno = frame.f_lineno
        self.tb_lasti = frame.f_lasti
        self.tb_next = None