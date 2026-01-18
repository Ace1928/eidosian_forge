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
def _build_collect_only_summary_stats_line(self) -> Tuple[List[Tuple[str, Dict[str, bool]]], str]:
    deselected = len(self._get_reports_to_display('deselected'))
    errors = len(self._get_reports_to_display('error'))
    if self._numcollected == 0:
        parts = [('no tests collected', {'yellow': True})]
        main_color = 'yellow'
    elif deselected == 0:
        main_color = 'green'
        collected_output = '%d %s collected' % pluralize(self._numcollected, 'test')
        parts = [(collected_output, {main_color: True})]
    else:
        all_tests_were_deselected = self._numcollected == deselected
        if all_tests_were_deselected:
            main_color = 'yellow'
            collected_output = f'no tests collected ({deselected} deselected)'
        else:
            main_color = 'green'
            selected = self._numcollected - deselected
            collected_output = f'{selected}/{self._numcollected} tests collected ({deselected} deselected)'
        parts = [(collected_output, {main_color: True})]
    if errors:
        main_color = _color_for_type['error']
        parts += [('%d %s' % pluralize(errors, 'error'), {main_color: True})]
    return (parts, main_color)