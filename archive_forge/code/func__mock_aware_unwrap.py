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
def _mock_aware_unwrap(func: Callable[..., Any], *, stop: Optional[Callable[[Any], Any]]=None) -> Any:
    try:
        if stop is None or stop is _is_mocked:
            return real_unwrap(func, stop=_is_mocked)
        _stop = stop
        return real_unwrap(func, stop=lambda obj: _is_mocked(obj) or _stop(func))
    except Exception as e:
        warnings.warn("Got %r when unwrapping %r.  This is usually caused by a violation of Python's object protocol; see e.g. https://github.com/pytest-dev/pytest/issues/5080" % (e, func), PytestWarning)
        raise