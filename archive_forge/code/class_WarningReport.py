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
@dataclasses.dataclass
class WarningReport:
    """Simple structure to hold warnings information captured by ``pytest_warning_recorded``.

    :ivar str message:
        User friendly message about the warning.
    :ivar str|None nodeid:
        nodeid that generated the warning (see ``get_location``).
    :ivar tuple fslocation:
        File system location of the source of the warning (see ``get_location``).
    """
    message: str
    nodeid: Optional[str] = None
    fslocation: Optional[Tuple[str, int]] = None
    count_towards_summary: ClassVar = True

    def get_location(self, config: Config) -> Optional[str]:
        """Return the more user-friendly information about the location of a warning, or None."""
        if self.nodeid:
            return self.nodeid
        if self.fslocation:
            filename, linenum = self.fslocation
            relpath = bestrelpath(config.invocation_params.dir, absolutepath(filename))
            return f'{relpath}:{linenum}'
        return None