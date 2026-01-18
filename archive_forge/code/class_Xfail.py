from collections.abc import Mapping
import dataclasses
import os
import platform
import sys
import traceback
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Type
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.mark.structures import Mark
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.reports import BaseReport
from _pytest.reports import TestReport
from _pytest.runner import CallInfo
from _pytest.stash import StashKey
@dataclasses.dataclass(frozen=True)
class Xfail:
    """The result of evaluate_xfail_marks()."""
    __slots__ = ('reason', 'run', 'strict', 'raises')
    reason: str
    run: bool
    strict: bool
    raises: Optional[Tuple[Type[BaseException], ...]]