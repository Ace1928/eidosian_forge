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
class FixtureManager:
    """pytest fixture definitions and information is stored and managed
    from this class.

    During collection fm.parsefactories() is called multiple times to parse
    fixture function definitions into FixtureDef objects and internal
    data structures.

    During collection of test functions, metafunc-mechanics instantiate
    a FuncFixtureInfo object which is cached per node/func-name.
    This FuncFixtureInfo object is later retrieved by Function nodes
    which themselves offer a fixturenames attribute.

    The FuncFixtureInfo object holds information about fixtures and FixtureDefs
    relevant for a particular function. An initial list of fixtures is
    assembled like this:

    - ini-defined usefixtures
    - autouse-marked fixtures along the collection chain up from the function
    - usefixtures markers at module/class/function level
    - test function funcargs

    Subsequently the funcfixtureinfo.fixturenames attribute is computed
    as the closure of the fixtures needed to setup the initial fixtures,
    i.e. fixtures needed by fixture functions themselves are appended
    to the fixturenames list.

    Upon the test-setup phases all fixturenames are instantiated, retrieved
    by a lookup of their FuncFixtureInfo.
    """

    def __init__(self, session: 'Session') -> None:
        self.session = session
        self.config: Config = session.config
        self._arg2fixturedefs: Final[Dict[str, List[FixtureDef[Any]]]] = {}
        self._holderobjseen: Final[Set[object]] = set()
        self._nodeid_autousenames: Final[Dict[str, List[str]]] = {'': self.config.getini('usefixtures')}
        session.config.pluginmanager.register(self, 'funcmanage')

    def getfixtureinfo(self, node: nodes.Item, func: Optional[Callable[..., object]], cls: Optional[type]) -> FuncFixtureInfo:
        """Calculate the :class:`FuncFixtureInfo` for an item.

        If ``func`` is None, or if the item sets an attribute
        ``nofuncargs = True``, then ``func`` is not examined at all.

        :param node:
            The item requesting the fixtures.
        :param func:
            The item's function.
        :param cls:
            If the function is a method, the method's class.
        """
        if func is not None and (not getattr(node, 'nofuncargs', False)):
            argnames = getfuncargnames(func, name=node.name, cls=cls)
        else:
            argnames = ()
        usefixturesnames = self._getusefixturesnames(node)
        autousenames = self._getautousenames(node)
        initialnames = deduplicate_names(autousenames, usefixturesnames, argnames)
        direct_parametrize_args = _get_direct_parametrize_args(node)
        names_closure, arg2fixturedefs = self.getfixtureclosure(parentnode=node, initialnames=initialnames, ignore_args=direct_parametrize_args)
        return FuncFixtureInfo(argnames, initialnames, names_closure, arg2fixturedefs)

    def pytest_plugin_registered(self, plugin: _PluggyPlugin, plugin_name: str) -> None:
        if plugin_name and plugin_name.endswith('conftest.py'):
            conftestpath = absolutepath(plugin_name)
            try:
                nodeid = str(conftestpath.parent.relative_to(self.config.rootpath))
            except ValueError:
                nodeid = ''
            if nodeid == '.':
                nodeid = ''
            if os.sep != nodes.SEP:
                nodeid = nodeid.replace(os.sep, nodes.SEP)
        else:
            nodeid = None
        self.parsefactories(plugin, nodeid)

    def _getautousenames(self, node: nodes.Node) -> Iterator[str]:
        """Return the names of autouse fixtures applicable to node."""
        for parentnode in node.listchain():
            basenames = self._nodeid_autousenames.get(parentnode.nodeid)
            if basenames:
                yield from basenames

    def _getusefixturesnames(self, node: nodes.Item) -> Iterator[str]:
        """Return the names of usefixtures fixtures applicable to node."""
        for mark in node.iter_markers(name='usefixtures'):
            yield from mark.args

    def getfixtureclosure(self, parentnode: nodes.Node, initialnames: Tuple[str, ...], ignore_args: AbstractSet[str]) -> Tuple[List[str], Dict[str, Sequence[FixtureDef[Any]]]]:
        fixturenames_closure = list(initialnames)
        arg2fixturedefs: Dict[str, Sequence[FixtureDef[Any]]] = {}
        lastlen = -1
        while lastlen != len(fixturenames_closure):
            lastlen = len(fixturenames_closure)
            for argname in fixturenames_closure:
                if argname in ignore_args:
                    continue
                if argname in arg2fixturedefs:
                    continue
                fixturedefs = self.getfixturedefs(argname, parentnode)
                if fixturedefs:
                    arg2fixturedefs[argname] = fixturedefs
                    for arg in fixturedefs[-1].argnames:
                        if arg not in fixturenames_closure:
                            fixturenames_closure.append(arg)

        def sort_by_scope(arg_name: str) -> Scope:
            try:
                fixturedefs = arg2fixturedefs[arg_name]
            except KeyError:
                return Scope.Function
            else:
                return fixturedefs[-1]._scope
        fixturenames_closure.sort(key=sort_by_scope, reverse=True)
        return (fixturenames_closure, arg2fixturedefs)

    def pytest_generate_tests(self, metafunc: 'Metafunc') -> None:
        """Generate new tests based on parametrized fixtures used by the given metafunc"""

        def get_parametrize_mark_argnames(mark: Mark) -> Sequence[str]:
            args, _ = ParameterSet._parse_parametrize_args(*mark.args, **mark.kwargs)
            return args
        for argname in metafunc.fixturenames:
            fixture_defs = metafunc._arg2fixturedefs.get(argname)
            if not fixture_defs:
                continue
            if any((argname in get_parametrize_mark_argnames(mark) for mark in metafunc.definition.iter_markers('parametrize'))):
                continue
            for fixturedef in reversed(fixture_defs):
                if fixturedef.params is not None:
                    metafunc.parametrize(argname, fixturedef.params, indirect=True, scope=fixturedef.scope, ids=fixturedef.ids)
                    break
                if argname not in fixturedef.argnames:
                    break

    def pytest_collection_modifyitems(self, items: List[nodes.Item]) -> None:
        items[:] = reorder_items(items)

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

    @overload
    def parsefactories(self, node_or_obj: nodes.Node, *, unittest: bool=...) -> None:
        raise NotImplementedError()

    @overload
    def parsefactories(self, node_or_obj: object, nodeid: Optional[str], *, unittest: bool=...) -> None:
        raise NotImplementedError()

    def parsefactories(self, node_or_obj: Union[nodes.Node, object], nodeid: Union[str, NotSetType, None]=NOTSET, *, unittest: bool=False) -> None:
        """Collect fixtures from a collection node or object.

        Found fixtures are parsed into `FixtureDef`s and saved.

        If `node_or_object` is a collection node (with an underlying Python
        object), the node's object is traversed and the node's nodeid is used to
        determine the fixtures' visibilty. `nodeid` must not be specified in
        this case.

        If `node_or_object` is an object (e.g. a plugin), the object is
        traversed and the given `nodeid` is used to determine the fixtures'
        visibility. `nodeid` must be specified in this case; None and "" mean
        total visibility.
        """
        if nodeid is not NOTSET:
            holderobj = node_or_obj
        else:
            assert isinstance(node_or_obj, nodes.Node)
            holderobj = cast(object, node_or_obj.obj)
            assert isinstance(node_or_obj.nodeid, str)
            nodeid = node_or_obj.nodeid
        if holderobj in self._holderobjseen:
            return
        self._holderobjseen.add(holderobj)
        for name in dir(holderobj):
            obj = safe_getattr(holderobj, name, None)
            marker = getfixturemarker(obj)
            if not isinstance(marker, FixtureFunctionMarker):
                continue
            if marker.name:
                name = marker.name
            func = get_real_method(obj, holderobj)
            self._register_fixture(name=name, nodeid=nodeid, func=func, scope=marker.scope, params=marker.params, unittest=unittest, ids=marker.ids, autouse=marker.autouse)

    def getfixturedefs(self, argname: str, node: nodes.Node) -> Optional[Sequence[FixtureDef[Any]]]:
        """Get FixtureDefs for a fixture name which are applicable
        to a given node.

        Returns None if there are no fixtures at all defined with the given
        name. (This is different from the case in which there are fixtures
        with the given name, but none applicable to the node. In this case,
        an empty result is returned).

        :param argname: Name of the fixture to search for.
        :param node: The requesting Node.
        """
        try:
            fixturedefs = self._arg2fixturedefs[argname]
        except KeyError:
            return None
        return tuple(self._matchfactories(fixturedefs, node))

    def _matchfactories(self, fixturedefs: Iterable[FixtureDef[Any]], node: nodes.Node) -> Iterator[FixtureDef[Any]]:
        parentnodeids = {n.nodeid for n in node.iter_parents()}
        for fixturedef in fixturedefs:
            if fixturedef.baseid in parentnodeids:
                yield fixturedef