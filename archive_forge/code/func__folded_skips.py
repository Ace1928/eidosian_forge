import argparse
from collections import Counter
import dataclasses
import datetime
from functools import partial
import inspect
from pathlib import Path
import platform
import sys
import textwrap
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import final
from typing import Generator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
from _pytest import timing
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._io import TerminalWriter
from _pytest._io.wcwidth import wcswidth
import _pytest._version
from _pytest.assertion.util import running_on_ci
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
def _folded_skips(startpath: Path, skipped: Sequence[CollectReport]) -> List[Tuple[int, str, Optional[int], str]]:
    d: Dict[Tuple[str, Optional[int], str], List[CollectReport]] = {}
    for event in skipped:
        assert event.longrepr is not None
        assert isinstance(event.longrepr, tuple), (event, event.longrepr)
        assert len(event.longrepr) == 3, (event, event.longrepr)
        fspath, lineno, reason = event.longrepr
        fspath = bestrelpath(startpath, Path(fspath))
        keywords = getattr(event, 'keywords', {})
        if event.when == 'setup' and 'skip' in keywords and ('pytestmark' not in keywords):
            key: Tuple[str, Optional[int], str] = (fspath, None, reason)
        else:
            key = (fspath, lineno, reason)
        d.setdefault(key, []).append(event)
    values: List[Tuple[int, str, Optional[int], str]] = []
    for key, events in d.items():
        values.append((len(events), *key))
    return values