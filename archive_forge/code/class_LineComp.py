import collections.abc
import contextlib
from fnmatch import fnmatch
import gc
import importlib
from io import StringIO
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
class LineComp:

    def __init__(self) -> None:
        self.stringio = StringIO()
        ':class:`python:io.StringIO()` instance used for input.'

    def assert_contains_lines(self, lines2: Sequence[str]) -> None:
        """Assert that ``lines2`` are contained (linearly) in :attr:`stringio`'s value.

        Lines are matched using :func:`LineMatcher.fnmatch_lines <pytest.LineMatcher.fnmatch_lines>`.
        """
        __tracebackhide__ = True
        val = self.stringio.getvalue()
        self.stringio.truncate(0)
        self.stringio.seek(0)
        lines1 = val.split('\n')
        LineMatcher(lines1).fnmatch_lines(lines2)