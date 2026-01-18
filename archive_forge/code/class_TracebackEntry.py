import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
class TracebackEntry:
    """A single entry in a Traceback."""
    __slots__ = ('_rawentry', '_repr_style')

    def __init__(self, rawentry: TracebackType, repr_style: Optional['Literal["short", "long"]']=None) -> None:
        self._rawentry: 'Final' = rawentry
        self._repr_style: 'Final' = repr_style

    def with_repr_style(self, repr_style: Optional['Literal["short", "long"]']) -> 'TracebackEntry':
        return TracebackEntry(self._rawentry, repr_style)

    @property
    def lineno(self) -> int:
        return self._rawentry.tb_lineno - 1

    @property
    def frame(self) -> Frame:
        return Frame(self._rawentry.tb_frame)

    @property
    def relline(self) -> int:
        return self.lineno - self.frame.code.firstlineno

    def __repr__(self) -> str:
        return '<TracebackEntry %s:%d>' % (self.frame.code.path, self.lineno + 1)

    @property
    def statement(self) -> 'Source':
        """_pytest._code.Source object for the current statement."""
        source = self.frame.code.fullsource
        assert source is not None
        return source.getstatement(self.lineno)

    @property
    def path(self) -> Union[Path, str]:
        """Path to the source code."""
        return self.frame.code.path

    @property
    def locals(self) -> Dict[str, Any]:
        """Locals of underlying frame."""
        return self.frame.f_locals

    def getfirstlinesource(self) -> int:
        return self.frame.code.firstlineno

    def getsource(self, astcache: Optional[Dict[Union[str, Path], ast.AST]]=None) -> Optional['Source']:
        """Return failing source code."""
        source = self.frame.code.fullsource
        if source is None:
            return None
        key = astnode = None
        if astcache is not None:
            key = self.frame.code.path
            if key is not None:
                astnode = astcache.get(key, None)
        start = self.getfirstlinesource()
        try:
            astnode, _, end = getstatementrange_ast(self.lineno, source, astnode=astnode)
        except SyntaxError:
            end = self.lineno + 1
        else:
            if key is not None and astcache is not None:
                astcache[key] = astnode
        return source[start:end]
    source = property(getsource)

    def ishidden(self, excinfo: Optional['ExceptionInfo[BaseException]']) -> bool:
        """Return True if the current frame has a var __tracebackhide__
        resolving to True.

        If __tracebackhide__ is a callable, it gets called with the
        ExceptionInfo instance and can decide whether to hide the traceback.

        Mostly for internal use.
        """
        tbh: Union[bool, Callable[[Optional[ExceptionInfo[BaseException]]], bool]] = False
        for maybe_ns_dct in (self.frame.f_locals, self.frame.f_globals):
            try:
                tbh = maybe_ns_dct['__tracebackhide__']
            except Exception:
                pass
            else:
                break
        if tbh and callable(tbh):
            return tbh(excinfo)
        return tbh

    def __str__(self) -> str:
        name = self.frame.code.name
        try:
            line = str(self.statement).lstrip()
        except KeyboardInterrupt:
            raise
        except BaseException:
            line = '???'
        return '  File %r:%d in %s\n  %s\n' % (str(self.path), self.lineno + 1, name, line)

    @property
    def name(self) -> str:
        """co_name of underlying code."""
        return self.frame.code.raw.co_name