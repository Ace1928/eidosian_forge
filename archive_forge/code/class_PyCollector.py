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
class PyCollector(PyobjMixin, nodes.Collector, abc.ABC):

    def funcnamefilter(self, name: str) -> bool:
        return self._matches_prefix_or_glob_option('python_functions', name)

    def isnosetest(self, obj: object) -> bool:
        """Look for the __test__ attribute, which is applied by the
        @nose.tools.istest decorator.
        """
        return safe_getattr(obj, '__test__', False) is True

    def classnamefilter(self, name: str) -> bool:
        return self._matches_prefix_or_glob_option('python_classes', name)

    def istestfunction(self, obj: object, name: str) -> bool:
        if self.funcnamefilter(name) or self.isnosetest(obj):
            if isinstance(obj, (staticmethod, classmethod)):
                obj = safe_getattr(obj, '__func__', False)
            return callable(obj) and fixtures.getfixturemarker(obj) is None
        else:
            return False

    def istestclass(self, obj: object, name: str) -> bool:
        return self.classnamefilter(name) or self.isnosetest(obj)

    def _matches_prefix_or_glob_option(self, option_name: str, name: str) -> bool:
        """Check if the given name matches the prefix or glob-pattern defined
        in ini configuration."""
        for option in self.config.getini(option_name):
            if name.startswith(option):
                return True
            elif ('*' in option or '?' in option or '[' in option) and fnmatch.fnmatch(name, option):
                return True
        return False

    def collect(self) -> Iterable[Union[nodes.Item, nodes.Collector]]:
        if not getattr(self.obj, '__test__', True):
            return []
        dicts = [getattr(self.obj, '__dict__', {})]
        if isinstance(self.obj, type):
            for basecls in self.obj.__mro__:
                dicts.append(basecls.__dict__)
        seen: Set[str] = set()
        dict_values: List[List[Union[nodes.Item, nodes.Collector]]] = []
        ihook = self.ihook
        for dic in dicts:
            values: List[Union[nodes.Item, nodes.Collector]] = []
            for name, obj in list(dic.items()):
                if name in IGNORED_ATTRIBUTES:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                res = ihook.pytest_pycollect_makeitem(collector=self, name=name, obj=obj)
                if res is None:
                    continue
                elif isinstance(res, list):
                    values.extend(res)
                else:
                    values.append(res)
            dict_values.append(values)
        result = []
        for values in reversed(dict_values):
            result.extend(values)
        return result

    def _genfunctions(self, name: str, funcobj) -> Iterator['Function']:
        modulecol = self.getparent(Module)
        assert modulecol is not None
        module = modulecol.obj
        clscol = self.getparent(Class)
        cls = clscol and clscol.obj or None
        definition = FunctionDefinition.from_parent(self, name=name, callobj=funcobj)
        fixtureinfo = definition._fixtureinfo
        metafunc = Metafunc(definition=definition, fixtureinfo=fixtureinfo, config=self.config, cls=cls, module=module, _ispytest=True)
        methods = []
        if hasattr(module, 'pytest_generate_tests'):
            methods.append(module.pytest_generate_tests)
        if cls is not None and hasattr(cls, 'pytest_generate_tests'):
            methods.append(cls().pytest_generate_tests)
        self.ihook.pytest_generate_tests.call_extra(methods, dict(metafunc=metafunc))
        if not metafunc._calls:
            yield Function.from_parent(self, name=name, fixtureinfo=fixtureinfo)
        else:
            fixtureinfo.prune_dependency_tree()
            for callspec in metafunc._calls:
                subname = f'{name}[{callspec.id}]'
                yield Function.from_parent(self, name=subname, callspec=callspec, fixtureinfo=fixtureinfo, keywords={callspec.id: True}, originalname=name)