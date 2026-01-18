import bdb
from contextlib import contextmanager
import functools
import inspect
import os
from pathlib import Path
import platform
import sys
import traceback
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import fixture
from _pytest.fixtures import TopRequest
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import skip
from _pytest.pathlib import fnmatch_ex
from _pytest.python import Module
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
class DoctestModule(Module):

    def collect(self) -> Iterable[DoctestItem]:
        import doctest

        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct
                line number is returned. This will be reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, 'fget', obj)
                if hasattr(obj, '__wrapped__'):
                    obj = inspect.unwrap(obj)
                return super()._find_lineno(obj, source_lines)

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    super()._find(tests, obj, name, module, source_lines, globs, seen)
            if sys.version_info < (3, 13):

                def _from_module(self, module, object):
                    """`cached_property` objects are never considered a part
                    of the 'current module'. As such they are skipped by doctest.
                    Here we override `_from_module` to check the underlying
                    function instead. https://github.com/python/cpython/issues/107995
                    """
                    if isinstance(object, functools.cached_property):
                        object = object.func
                    return super()._from_module(module, object)
            else:
                pass
        try:
            module = self.obj
        except Collector.CollectError:
            if self.config.getvalue('doctest_ignore_import_errors'):
                skip('unable to import module %r' % self.path)
            else:
                raise
        self.session._fixturemanager.parsefactories(self)
        finder = MockAwareDocTestFinder()
        optionflags = get_optionflags(self.config)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        for test in finder.find(module, module.__name__):
            if test.examples:
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)