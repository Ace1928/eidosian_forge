import abc
from collections import defaultdict
from collections import deque
import dataclasses
import functools
import inspect
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import _pytest
from _pytest import nodes
from _pytest._code import getfslineno
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import _PytestWrapper
from _pytest.compat import assert_never
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.compat import safe_getattr
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.deprecated import YIELD_FIXTURE
from _pytest.mark import Mark
from _pytest.mark import ParameterSet
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import TEST_OUTCOME
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.scope import _ScopeName
from _pytest.scope import HIGH_SCOPES
from _pytest.scope import Scope
@final
class FixtureLookupError(LookupError):
    """Could not return a requested fixture (missing or invalid)."""

    def __init__(self, argname: Optional[str], request: FixtureRequest, msg: Optional[str]=None) -> None:
        self.argname = argname
        self.request = request
        self.fixturestack = request._get_fixturestack()
        self.msg = msg

    def formatrepr(self) -> 'FixtureLookupErrorRepr':
        tblines: List[str] = []
        addline = tblines.append
        stack = [self.request._pyfuncitem.obj]
        stack.extend(map(lambda x: x.func, self.fixturestack))
        msg = self.msg
        if msg is not None:
            stack = stack[:-1]
        for function in stack:
            fspath, lineno = getfslineno(function)
            try:
                lines, _ = inspect.getsourcelines(get_real_func(function))
            except (OSError, IndexError, TypeError):
                error_msg = 'file %s, line %s: source code not available'
                addline(error_msg % (fspath, lineno + 1))
            else:
                addline(f'file {fspath}, line {lineno + 1}')
                for i, line in enumerate(lines):
                    line = line.rstrip()
                    addline('  ' + line)
                    if line.lstrip().startswith('def'):
                        break
        if msg is None:
            fm = self.request._fixturemanager
            available = set()
            parent = self.request._pyfuncitem.parent
            assert parent is not None
            for name, fixturedefs in fm._arg2fixturedefs.items():
                faclist = list(fm._matchfactories(fixturedefs, parent))
                if faclist:
                    available.add(name)
            if self.argname in available:
                msg = f" recursive dependency involving fixture '{self.argname}' detected"
            else:
                msg = f"fixture '{self.argname}' not found"
            msg += '\n available fixtures: {}'.format(', '.join(sorted(available)))
            msg += "\n use 'pytest --fixtures [testpath]' for help on them."
        return FixtureLookupErrorRepr(fspath, lineno, tblines, msg, self.argname)