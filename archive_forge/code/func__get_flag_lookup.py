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
def _get_flag_lookup() -> Dict[str, int]:
    import doctest
    return dict(DONT_ACCEPT_TRUE_FOR_1=doctest.DONT_ACCEPT_TRUE_FOR_1, DONT_ACCEPT_BLANKLINE=doctest.DONT_ACCEPT_BLANKLINE, NORMALIZE_WHITESPACE=doctest.NORMALIZE_WHITESPACE, ELLIPSIS=doctest.ELLIPSIS, IGNORE_EXCEPTION_DETAIL=doctest.IGNORE_EXCEPTION_DETAIL, COMPARISON_FLAGS=doctest.COMPARISON_FLAGS, ALLOW_UNICODE=_get_allow_unicode_flag(), ALLOW_BYTES=_get_allow_bytes_flag(), NUMBER=_get_number_flag())