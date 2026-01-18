import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
class IPDoctestModule(pytest.Module):

    def collect(self) -> Iterable[IPDoctestItem]:
        import doctest
        from .ipdoctest import DocTestFinder, IPDocTestParser

        class MockAwareDocTestFinder(DocTestFinder):
            """A hackish ipdoctest finder that overrides stdlib internals to fix a stdlib bug.

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
        if self.path.name == 'conftest.py':
            if int(pytest.__version__.split('.')[0]) < 7:
                module = self.config.pluginmanager._importconftest(self.path, self.config.getoption('importmode'))
            else:
                module = self.config.pluginmanager._importconftest(self.path, self.config.getoption('importmode'), rootpath=self.config.rootpath)
        else:
            try:
                module = import_path(self.path, root=self.config.rootpath)
            except ImportError:
                if self.config.getvalue('ipdoctest_ignore_import_errors'):
                    pytest.skip('unable to import module %r' % self.path)
                else:
                    raise
        finder = MockAwareDocTestFinder(parser=IPDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        for test in finder.find(module, module.__name__):
            if test.examples:
                yield IPDoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
    if int(pytest.__version__.split('.')[0]) < 7:

        @property
        def path(self) -> Path:
            return Path(self.fspath)

        @classmethod
        def from_parent(cls, parent, *, fspath=None, path: Optional[Path]=None, **kw):
            if path is not None:
                import py.path
                fspath = py.path.local(path)
            return super().from_parent(parent=parent, fspath=fspath, **kw)