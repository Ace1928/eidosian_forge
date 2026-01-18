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
def _remove_unwanted_precision(self, want: str, got: str) -> str:
    wants = list(self._number_re.finditer(want))
    gots = list(self._number_re.finditer(got))
    if len(wants) != len(gots):
        return got
    offset = 0
    for w, g in zip(wants, gots):
        fraction: Optional[str] = w.group('fraction')
        exponent: Optional[str] = w.group('exponent1')
        if exponent is None:
            exponent = w.group('exponent2')
        precision = 0 if fraction is None else len(fraction)
        if exponent is not None:
            precision -= int(exponent)
        if float(w.group()) == approx(float(g.group()), abs=10 ** (-precision)):
            got = got[:g.start() + offset] + w.group() + got[g.end() + offset:]
            offset += w.end() - w.start() - (g.end() - g.start())
    return got