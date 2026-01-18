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
@final
@dataclasses.dataclass
class TempdirFactory:
    """Backward compatibility wrapper that implements ``py.path.local``
    for :class:`TempPathFactory`.

    .. note::
        These days, it is preferred to use ``tmp_path_factory``.

        :ref:`About the tmpdir and tmpdir_factory fixtures<tmpdir and tmpdir_factory>`.

    """
    _tmppath_factory: TempPathFactory

    def __init__(self, tmppath_factory: TempPathFactory, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self._tmppath_factory = tmppath_factory

    def mktemp(self, basename: str, numbered: bool=True) -> LEGACY_PATH:
        """Same as :meth:`TempPathFactory.mktemp`, but returns a ``py.path.local`` object."""
        return legacy_path(self._tmppath_factory.mktemp(basename, numbered).resolve())

    def getbasetemp(self) -> LEGACY_PATH:
        """Same as :meth:`TempPathFactory.getbasetemp`, but returns a ``py.path.local`` object."""
        return legacy_path(self._tmppath_factory.getbasetemp().resolve())