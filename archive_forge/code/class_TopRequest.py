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
class TopRequest(FixtureRequest):
    """The type of the ``request`` fixture in a test function."""

    def __init__(self, pyfuncitem: 'Function', *, _ispytest: bool=False) -> None:
        super().__init__(fixturename=None, pyfuncitem=pyfuncitem, arg2fixturedefs=pyfuncitem._fixtureinfo.name2fixturedefs.copy(), fixture_defs={}, _ispytest=_ispytest)

    @property
    def _scope(self) -> Scope:
        return Scope.Function

    @property
    def node(self):
        return self._pyfuncitem

    def __repr__(self) -> str:
        return '<FixtureRequest for %r>' % self.node

    def _fillfixtures(self) -> None:
        item = self._pyfuncitem
        for argname in item.fixturenames:
            if argname not in item.funcargs:
                item.funcargs[argname] = self.getfixturevalue(argname)

    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        self.node.addfinalizer(finalizer)