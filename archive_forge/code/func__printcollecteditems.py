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
def _printcollecteditems(self, items: Sequence[Item]) -> None:
    test_cases_verbosity = self.config.get_verbosity(Config.VERBOSITY_TEST_CASES)
    if test_cases_verbosity < 0:
        if test_cases_verbosity < -1:
            counts = Counter((item.nodeid.split('::', 1)[0] for item in items))
            for name, count in sorted(counts.items()):
                self._tw.line('%s: %d' % (name, count))
        else:
            for item in items:
                self._tw.line(item.nodeid)
        return
    stack: List[Node] = []
    indent = ''
    for item in items:
        needed_collectors = item.listchain()[1:]
        while stack:
            if stack == needed_collectors[:len(stack)]:
                break
            stack.pop()
        for col in needed_collectors[len(stack):]:
            stack.append(col)
            indent = (len(stack) - 1) * '  '
            self._tw.line(f'{indent}{col}')
            if test_cases_verbosity >= 1:
                obj = getattr(col, 'obj', None)
                doc = inspect.getdoc(obj) if obj else None
                if doc:
                    for line in doc.splitlines():
                        self._tw.line('{}{}'.format(indent + '  ', line))