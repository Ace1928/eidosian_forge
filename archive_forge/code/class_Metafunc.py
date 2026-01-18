import abc
from collections import Counter
from collections import defaultdict
import dataclasses
import enum
import fnmatch
from functools import partial
import inspect
import itertools
import os
from pathlib import Path
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import saferepr
from _pytest.compat import ascii_escaped
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_async_function
from _pytest.compat import is_generator
from _pytest.compat import LEGACY_PATH
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import FuncFixtureInfo
from _pytest.fixtures import get_scope_node
from _pytest.main import Session
from _pytest.mark import MARK_GEN
from _pytest.mark import ParameterSet
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import scandir
from _pytest.scope import _ScopeName
from _pytest.scope import Scope
from _pytest.stash import StashKey
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestReturnNotNoneWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
@final
class Metafunc:
    """Objects passed to the :hook:`pytest_generate_tests` hook.

    They help to inspect a test function and to generate tests according to
    test configuration or values specified in the class or module where a
    test function is defined.
    """

    def __init__(self, definition: 'FunctionDefinition', fixtureinfo: fixtures.FuncFixtureInfo, config: Config, cls=None, module=None, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self.definition = definition
        self.config = config
        self.module = module
        self.function = definition.obj
        self.fixturenames = fixtureinfo.names_closure
        self.cls = cls
        self._arg2fixturedefs = fixtureinfo.name2fixturedefs
        self._calls: List[CallSpec2] = []

    def parametrize(self, argnames: Union[str, Sequence[str]], argvalues: Iterable[Union[ParameterSet, Sequence[object], object]], indirect: Union[bool, Sequence[str]]=False, ids: Optional[Union[Iterable[Optional[object]], Callable[[Any], Optional[object]]]]=None, scope: Optional[_ScopeName]=None, *, _param_mark: Optional[Mark]=None) -> None:
        """Add new invocations to the underlying test function using the list
        of argvalues for the given argnames. Parametrization is performed
        during the collection phase. If you need to setup expensive resources
        see about setting indirect to do it rather than at test setup time.

        Can be called multiple times per test function (but only on different
        argument names), in which case each call parametrizes all previous
        parametrizations, e.g.

        ::

            unparametrized:         t
            parametrize ["x", "y"]: t[x], t[y]
            parametrize [1, 2]:     t[x-1], t[x-2], t[y-1], t[y-2]

        :param argnames:
            A comma-separated string denoting one or more argument names, or
            a list/tuple of argument strings.

        :param argvalues:
            The list of argvalues determines how often a test is invoked with
            different argument values.

            If only one argname was specified argvalues is a list of values.
            If N argnames were specified, argvalues must be a list of
            N-tuples, where each tuple-element specifies a value for its
            respective argname.

        :param indirect:
            A list of arguments' names (subset of argnames) or a boolean.
            If True the list contains all names from the argnames. Each
            argvalue corresponding to an argname in this list will
            be passed as request.param to its respective argname fixture
            function so that it can perform more expensive setups during the
            setup phase of a test rather than at collection time.

        :param ids:
            Sequence of (or generator for) ids for ``argvalues``,
            or a callable to return part of the id for each argvalue.

            With sequences (and generators like ``itertools.count()``) the
            returned ids should be of type ``string``, ``int``, ``float``,
            ``bool``, or ``None``.
            They are mapped to the corresponding index in ``argvalues``.
            ``None`` means to use the auto-generated id.

            If it is a callable it will be called for each entry in
            ``argvalues``, and the return value is used as part of the
            auto-generated id for the whole set (where parts are joined with
            dashes ("-")).
            This is useful to provide more specific ids for certain items, e.g.
            dates.  Returning ``None`` will use an auto-generated id.

            If no ids are provided they will be generated automatically from
            the argvalues.

        :param scope:
            If specified it denotes the scope of the parameters.
            The scope is used for grouping tests by parameter instances.
            It will also override any fixture-function defined scope, allowing
            to set a dynamic scope using test context or configuration.
        """
        argnames, parametersets = ParameterSet._for_parametrize(argnames, argvalues, self.function, self.config, nodeid=self.definition.nodeid)
        del argvalues
        if 'request' in argnames:
            fail("'request' is a reserved name and cannot be used in @pytest.mark.parametrize", pytrace=False)
        if scope is not None:
            scope_ = Scope.from_user(scope, descr=f'parametrize() call in {self.function.__name__}')
        else:
            scope_ = _find_parametrized_scope(argnames, self._arg2fixturedefs, indirect)
        self._validate_if_using_arg_names(argnames, indirect)
        if _param_mark and _param_mark._param_ids_from:
            generated_ids = _param_mark._param_ids_from._param_ids_generated
            if generated_ids is not None:
                ids = generated_ids
        ids = self._resolve_parameter_set_ids(argnames, ids, parametersets, nodeid=self.definition.nodeid)
        if _param_mark and _param_mark._param_ids_from and (generated_ids is None):
            object.__setattr__(_param_mark._param_ids_from, '_param_ids_generated', ids)
        node = None
        if scope_ is not Scope.Function:
            collector = self.definition.parent
            assert collector is not None
            node = get_scope_node(collector, scope_)
            if node is None:
                if scope_ is Scope.Class:
                    assert isinstance(collector, Module)
                    node = collector
                elif scope_ is Scope.Package:
                    node = collector.session
                else:
                    assert False, f'Unhandled missing scope: {scope}'
        if node is None:
            name2pseudofixturedef = None
        else:
            default: Dict[str, FixtureDef[Any]] = {}
            name2pseudofixturedef = node.stash.setdefault(name2pseudofixturedef_key, default)
        arg_directness = self._resolve_args_directness(argnames, indirect)
        for argname in argnames:
            if arg_directness[argname] == 'indirect':
                continue
            if name2pseudofixturedef is not None and argname in name2pseudofixturedef:
                fixturedef = name2pseudofixturedef[argname]
            else:
                fixturedef = FixtureDef(config=self.config, baseid='', argname=argname, func=get_direct_param_fixture_func, scope=scope_, params=None, unittest=False, ids=None, _ispytest=True)
                if name2pseudofixturedef is not None:
                    name2pseudofixturedef[argname] = fixturedef
            self._arg2fixturedefs[argname] = [fixturedef]
        newcalls = []
        for callspec in self._calls or [CallSpec2()]:
            for param_index, (param_id, param_set) in enumerate(zip(ids, parametersets)):
                newcallspec = callspec.setmulti(argnames=argnames, valset=param_set.values, id=param_id, marks=param_set.marks, scope=scope_, param_index=param_index)
                newcalls.append(newcallspec)
        self._calls = newcalls

    def _resolve_parameter_set_ids(self, argnames: Sequence[str], ids: Optional[Union[Iterable[Optional[object]], Callable[[Any], Optional[object]]]], parametersets: Sequence[ParameterSet], nodeid: str) -> List[str]:
        """Resolve the actual ids for the given parameter sets.

        :param argnames:
            Argument names passed to ``parametrize()``.
        :param ids:
            The `ids` parameter of the ``parametrize()`` call (see docs).
        :param parametersets:
            The parameter sets, each containing a set of values corresponding
            to ``argnames``.
        :param nodeid str:
            The nodeid of the definition item that generated this
            parametrization.
        :returns:
            List with ids for each parameter set given.
        """
        if ids is None:
            idfn = None
            ids_ = None
        elif callable(ids):
            idfn = ids
            ids_ = None
        else:
            idfn = None
            ids_ = self._validate_ids(ids, parametersets, self.function.__name__)
        id_maker = IdMaker(argnames, parametersets, idfn, ids_, self.config, nodeid=nodeid, func_name=self.function.__name__)
        return id_maker.make_unique_parameterset_ids()

    def _validate_ids(self, ids: Iterable[Optional[object]], parametersets: Sequence[ParameterSet], func_name: str) -> List[Optional[object]]:
        try:
            num_ids = len(ids)
        except TypeError:
            try:
                iter(ids)
            except TypeError as e:
                raise TypeError('ids must be a callable or an iterable') from e
            num_ids = len(parametersets)
        if num_ids != len(parametersets) and num_ids != 0:
            msg = 'In {}: {} parameter sets specified, with different number of ids: {}'
            fail(msg.format(func_name, len(parametersets), num_ids), pytrace=False)
        return list(itertools.islice(ids, num_ids))

    def _resolve_args_directness(self, argnames: Sequence[str], indirect: Union[bool, Sequence[str]]) -> Dict[str, Literal['indirect', 'direct']]:
        """Resolve if each parametrized argument must be considered an indirect
        parameter to a fixture of the same name, or a direct parameter to the
        parametrized function, based on the ``indirect`` parameter of the
        parametrized() call.

        :param argnames:
            List of argument names passed to ``parametrize()``.
        :param indirect:
            Same as the ``indirect`` parameter of ``parametrize()``.
        :returns
            A dict mapping each arg name to either "indirect" or "direct".
        """
        arg_directness: Dict[str, Literal['indirect', 'direct']]
        if isinstance(indirect, bool):
            arg_directness = dict.fromkeys(argnames, 'indirect' if indirect else 'direct')
        elif isinstance(indirect, Sequence):
            arg_directness = dict.fromkeys(argnames, 'direct')
            for arg in indirect:
                if arg not in argnames:
                    fail(f"In {self.function.__name__}: indirect fixture '{arg}' doesn't exist", pytrace=False)
                arg_directness[arg] = 'indirect'
        else:
            fail(f'In {self.function.__name__}: expected Sequence or boolean for indirect, got {type(indirect).__name__}', pytrace=False)
        return arg_directness

    def _validate_if_using_arg_names(self, argnames: Sequence[str], indirect: Union[bool, Sequence[str]]) -> None:
        """Check if all argnames are being used, by default values, or directly/indirectly.

        :param List[str] argnames: List of argument names passed to ``parametrize()``.
        :param indirect: Same as the ``indirect`` parameter of ``parametrize()``.
        :raises ValueError: If validation fails.
        """
        default_arg_names = set(get_default_arg_names(self.function))
        func_name = self.function.__name__
        for arg in argnames:
            if arg not in self.fixturenames:
                if arg in default_arg_names:
                    fail(f"In {func_name}: function already takes an argument '{arg}' with a default value", pytrace=False)
                else:
                    if isinstance(indirect, Sequence):
                        name = 'fixture' if arg in indirect else 'argument'
                    else:
                        name = 'fixture' if indirect else 'argument'
                    fail(f"In {func_name}: function uses no {name} '{arg}'", pytrace=False)