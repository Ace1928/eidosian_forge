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
def _register_fixture(self, *, name: str, func: '_FixtureFunc[object]', nodeid: Optional[str], scope: Union[Scope, _ScopeName, Callable[[str, Config], _ScopeName], None]='function', params: Optional[Sequence[object]]=None, ids: Optional[Union[Tuple[Optional[object], ...], Callable[[Any], Optional[object]]]]=None, autouse: bool=False, unittest: bool=False) -> None:
    """Register a fixture

        :param name:
            The fixture's name.
        :param func:
            The fixture's implementation function.
        :param nodeid:
            The visibility of the fixture. The fixture will be available to the
            node with this nodeid and its children in the collection tree.
            None means that the fixture is visible to the entire collection tree,
            e.g. a fixture defined for general use in a plugin.
        :param scope:
            The fixture's scope.
        :param params:
            The fixture's parametrization params.
        :param ids:
            The fixture's IDs.
        :param autouse:
            Whether this is an autouse fixture.
        :param unittest:
            Set this if this is a unittest fixture.
        """
    fixture_def = FixtureDef(config=self.config, baseid=nodeid, argname=name, func=func, scope=scope, params=params, unittest=unittest, ids=ids, _ispytest=True)
    faclist = self._arg2fixturedefs.setdefault(name, [])
    if fixture_def.has_location:
        faclist.append(fixture_def)
    else:
        i = len([f for f in faclist if not f.has_location])
        faclist.insert(i, fixture_def)
    if autouse:
        self._nodeid_autousenames.setdefault(nodeid or '', []).append(name)