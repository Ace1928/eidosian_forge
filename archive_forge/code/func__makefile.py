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
def _makefile(self, ext: str, lines: Sequence[Union[Any, bytes]], files: Dict[str, str], encoding: str='utf-8') -> Path:
    items = list(files.items())
    if ext and (not ext.startswith('.')):
        raise ValueError(f'pytester.makefile expects a file extension, try .{ext} instead of {ext}')

    def to_text(s: Union[Any, bytes]) -> str:
        return s.decode(encoding) if isinstance(s, bytes) else str(s)
    if lines:
        source = '\n'.join((to_text(x) for x in lines))
        basename = self._name
        items.insert(0, (basename, source))
    ret = None
    for basename, value in items:
        p = self.path.joinpath(basename).with_suffix(ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        source_ = Source(value)
        source = '\n'.join((to_text(line) for line in source_.lines))
        p.write_text(source.strip(), encoding=encoding)
        if ret is None:
            ret = p
    assert ret is not None
    return ret