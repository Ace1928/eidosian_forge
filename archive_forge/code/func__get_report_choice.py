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
def _get_report_choice(key: str) -> int:
    """Return the actual `ipdoctest` module flag value.

    We want to do it as late as possible to avoid importing `ipdoctest` and all
    its dependencies when parsing options, as it adds overhead and breaks tests.
    """
    import doctest
    return {DOCTEST_REPORT_CHOICE_UDIFF: doctest.REPORT_UDIFF, DOCTEST_REPORT_CHOICE_CDIFF: doctest.REPORT_CDIFF, DOCTEST_REPORT_CHOICE_NDIFF: doctest.REPORT_NDIFF, DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE: doctest.REPORT_ONLY_FIRST_FAILURE, DOCTEST_REPORT_CHOICE_NONE: 0}[key]