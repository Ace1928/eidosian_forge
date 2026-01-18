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
class IPDoctestTextfile(pytest.Module):
    obj = None

    def collect(self) -> Iterable[IPDoctestItem]:
        import doctest
        from .ipdoctest import IPDocTestParser
        encoding = self.config.getini('ipdoctest_encoding')
        text = self.path.read_text(encoding)
        filename = str(self.path)
        name = self.path.name
        globs = {'__name__': '__main__'}
        optionflags = get_optionflags(self)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        parser = IPDocTestParser()
        test = parser.get_doctest(text, globs, name, filename, 0)
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