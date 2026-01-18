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
class FixtureRequest(abc.ABC):
    """The type of the ``request`` fixture.

    A request object gives access to the requesting test context and has a
    ``param`` attribute in case the fixture is parametrized.
    """

    def __init__(self, pyfuncitem: 'Function', fixturename: Optional[str], arg2fixturedefs: Dict[str, Sequence['FixtureDef[Any]']], fixture_defs: Dict[str, 'FixtureDef[Any]'], *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self.fixturename: Final = fixturename
        self._pyfuncitem: Final = pyfuncitem
        self._arg2fixturedefs: Final = arg2fixturedefs
        self._fixture_defs: Final = fixture_defs
        self.param: Any

    @property
    def _fixturemanager(self) -> 'FixtureManager':
        return self._pyfuncitem.session._fixturemanager

    @property
    @abc.abstractmethod
    def _scope(self) -> Scope:
        raise NotImplementedError()

    @property
    def scope(self) -> _ScopeName:
        """Scope string, one of "function", "class", "module", "package", "session"."""
        return self._scope.value

    @property
    def fixturenames(self) -> List[str]:
        """Names of all active fixtures in this request."""
        result = list(self._pyfuncitem.fixturenames)
        result.extend(set(self._fixture_defs).difference(result))
        return result

    @property
    @abc.abstractmethod
    def node(self):
        """Underlying collection node (depends on current request scope)."""
        raise NotImplementedError()

    def _getnextfixturedef(self, argname: str) -> 'FixtureDef[Any]':
        fixturedefs = self._arg2fixturedefs.get(argname, None)
        if fixturedefs is None:
            fixturedefs = self._fixturemanager.getfixturedefs(argname, self._pyfuncitem)
            if fixturedefs is not None:
                self._arg2fixturedefs[argname] = fixturedefs
        if fixturedefs is None:
            raise FixtureLookupError(argname, self)
        if not fixturedefs:
            raise FixtureLookupError(argname, self)
        index = -1
        for request in self._iter_chain():
            if request.fixturename == argname:
                index -= 1
        if -index > len(fixturedefs):
            raise FixtureLookupError(argname, self)
        return fixturedefs[index]

    @property
    def config(self) -> Config:
        """The pytest config object associated with this request."""
        return self._pyfuncitem.config

    @property
    def function(self):
        """Test function object if the request has a per-function scope."""
        if self.scope != 'function':
            raise AttributeError(f'function not available in {self.scope}-scoped context')
        return self._pyfuncitem.obj

    @property
    def cls(self):
        """Class (can be None) where the test function was collected."""
        if self.scope not in ('class', 'function'):
            raise AttributeError(f'cls not available in {self.scope}-scoped context')
        clscol = self._pyfuncitem.getparent(_pytest.python.Class)
        if clscol:
            return clscol.obj

    @property
    def instance(self):
        """Instance (can be None) on which test function was collected."""
        try:
            return self._pyfuncitem._testcase
        except AttributeError:
            function = getattr(self, 'function', None)
            return getattr(function, '__self__', None)

    @property
    def module(self):
        """Python module object where the test function was collected."""
        if self.scope not in ('function', 'class', 'module'):
            raise AttributeError(f'module not available in {self.scope}-scoped context')
        mod = self._pyfuncitem.getparent(_pytest.python.Module)
        assert mod is not None
        return mod.obj

    @property
    def path(self) -> Path:
        """Path where the test function was collected."""
        if self.scope not in ('function', 'class', 'module', 'package'):
            raise AttributeError(f'path not available in {self.scope}-scoped context')
        return self._pyfuncitem.path

    @property
    def keywords(self) -> MutableMapping[str, Any]:
        """Keywords/markers dictionary for the underlying node."""
        node: nodes.Node = self.node
        return node.keywords

    @property
    def session(self) -> 'Session':
        """Pytest session object."""
        return self._pyfuncitem.session

    @abc.abstractmethod
    def addfinalizer(self, finalizer: Callable[[], object]) -> None:
        """Add finalizer/teardown function to be called without arguments after
        the last test within the requesting test context finished execution."""
        raise NotImplementedError()

    def applymarker(self, marker: Union[str, MarkDecorator]) -> None:
        """Apply a marker to a single test function invocation.

        This method is useful if you don't want to have a keyword/marker
        on all function invocations.

        :param marker:
            An object created by a call to ``pytest.mark.NAME(...)``.
        """
        self.node.add_marker(marker)

    def raiseerror(self, msg: Optional[str]) -> NoReturn:
        """Raise a FixtureLookupError exception.

        :param msg:
            An optional custom error message.
        """
        raise FixtureLookupError(None, self, msg)

    def getfixturevalue(self, argname: str) -> Any:
        """Dynamically run a named fixture function.

        Declaring fixtures via function argument is recommended where possible.
        But if you can only decide whether to use another fixture at test
        setup time, you may use this function to retrieve it inside a fixture
        or test function body.

        This method can be used during the test setup phase or the test run
        phase, but during the test teardown phase a fixture's value may not
        be available.

        :param argname:
            The fixture name.
        :raises pytest.FixtureLookupError:
            If the given fixture could not be found.
        """
        fixturedef = self._get_active_fixturedef(argname)
        assert fixturedef.cached_result is not None, f'The fixture value for "{argname}" is not available.  This can happen when the fixture has already been torn down.'
        return fixturedef.cached_result[0]

    def _iter_chain(self) -> Iterator['SubRequest']:
        """Yield all SubRequests in the chain, from self up.

        Note: does *not* yield the TopRequest.
        """
        current = self
        while isinstance(current, SubRequest):
            yield current
            current = current._parent_request

    def _get_active_fixturedef(self, argname: str) -> Union['FixtureDef[object]', PseudoFixtureDef[object]]:
        fixturedef = self._fixture_defs.get(argname)
        if fixturedef is None:
            try:
                fixturedef = self._getnextfixturedef(argname)
            except FixtureLookupError:
                if argname == 'request':
                    cached_result = (self, [0], None)
                    return PseudoFixtureDef(cached_result, Scope.Function)
                raise
            self._compute_fixture_value(fixturedef)
            self._fixture_defs[argname] = fixturedef
        return fixturedef

    def _get_fixturestack(self) -> List['FixtureDef[Any]']:
        values = [request._fixturedef for request in self._iter_chain()]
        values.reverse()
        return values

    def _compute_fixture_value(self, fixturedef: 'FixtureDef[object]') -> None:
        """Create a SubRequest based on "self" and call the execute method
        of the given FixtureDef object.

        This will force the FixtureDef object to throw away any previous
        results and compute a new fixture value, which will be stored into
        the FixtureDef object itself.
        """
        argname = fixturedef.argname
        funcitem = self._pyfuncitem
        try:
            callspec = funcitem.callspec
        except AttributeError:
            callspec = None
        if callspec is not None and argname in callspec.params:
            param = callspec.params[argname]
            param_index = callspec.indices[argname]
            scope = callspec._arg2scope[argname]
        else:
            param = NOTSET
            param_index = 0
            scope = fixturedef._scope
            has_params = fixturedef.params is not None
            fixtures_not_supported = getattr(funcitem, 'nofuncargs', False)
            if has_params and fixtures_not_supported:
                msg = f'{funcitem.name} does not support fixtures, maybe unittest.TestCase subclass?\nNode id: {funcitem.nodeid}\nFunction type: {type(funcitem).__name__}'
                fail(msg, pytrace=False)
            if has_params:
                frame = inspect.stack()[3]
                frameinfo = inspect.getframeinfo(frame[0])
                source_path = absolutepath(frameinfo.filename)
                source_lineno = frameinfo.lineno
                try:
                    source_path_str = str(source_path.relative_to(funcitem.config.rootpath))
                except ValueError:
                    source_path_str = str(source_path)
                msg = "The requested fixture has no parameter defined for test:\n    {}\n\nRequested fixture '{}' defined in:\n{}\n\nRequested here:\n{}:{}".format(funcitem.nodeid, fixturedef.argname, getlocation(fixturedef.func, funcitem.config.rootpath), source_path_str, source_lineno)
                fail(msg, pytrace=False)
        subrequest = SubRequest(self, scope, param, param_index, fixturedef, _ispytest=True)
        subrequest._check_scope(argname, self._scope, scope)
        try:
            fixturedef.execute(request=subrequest)
        finally:
            self._schedule_finalizers(fixturedef, subrequest)

    def _schedule_finalizers(self, fixturedef: 'FixtureDef[object]', subrequest: 'SubRequest') -> None:
        finalizer = functools.partial(fixturedef.finish, request=subrequest)
        subrequest.node.addfinalizer(finalizer)