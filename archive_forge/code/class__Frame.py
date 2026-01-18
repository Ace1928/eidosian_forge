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
class _Frame:
    """
    A fake frame object, used by L{_Traceback}.

    @ivar f_code: fake L{code<types.CodeType>} object
    @ivar f_lineno: line number
    @ivar f_globals: fake f_globals dictionary (usually empty)
    @ivar f_locals: fake f_locals dictionary (usually empty)
    @ivar f_back: previous stack frame (towards the caller)
    """

    def __init__(self, frameinfo, back):
        """
        @param frameinfo: (methodname, filename, lineno, locals, globals)
        @param back: previous (older) stack frame
        @type back: C{frame}
        """
        name, filename, lineno, localz, globalz = frameinfo
        self.f_code = _Code(name, filename)
        self.f_lineno = lineno
        self.f_globals = dict(globalz or {})
        self.f_locals = dict(localz or {})
        self.f_back = back
        self.f_lasti = 0
        self.f_builtins = vars(builtins).copy()
        self.f_trace = None