import dataclasses
from pathlib import Path
import shlex
import subprocess
from typing import Final
from typing import final
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from iniconfig import SectionWrapper
from _pytest.cacheprovider import Cache
from _pytest.compat import LEGACY_PATH
from _pytest.compat import legacy_path
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pytester import HookRecorder
from _pytest.pytester import Pytester
from _pytest.pytester import RunResult
from _pytest.terminal import TerminalReporter
from _pytest.tmpdir import TempPathFactory
def TerminalReporter_startdir(self: TerminalReporter) -> LEGACY_PATH:
    """The directory from which pytest was invoked.

    Prefer to use ``startpath`` which is a :class:`pathlib.Path`.

    :type: LEGACY_PATH
    """
    return legacy_path(self.startpath)