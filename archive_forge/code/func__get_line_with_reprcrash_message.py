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
def _get_line_with_reprcrash_message(config: Config, rep: BaseReport, tw: TerminalWriter, word_markup: Dict[str, bool]) -> str:
    """Get summary line for a report, trying to add reprcrash message."""
    verbose_word = rep._get_verbose_word(config)
    word = tw.markup(verbose_word, **word_markup)
    node = _get_node_id_with_markup(tw, config, rep)
    line = f'{word} {node}'
    line_width = wcswidth(line)
    try:
        msg = rep.longrepr.reprcrash.message
    except AttributeError:
        pass
    else:
        if not running_on_ci():
            available_width = tw.fullwidth - line_width
            msg = _format_trimmed(' - {}', msg, available_width)
        else:
            msg = f' - {msg}'
        if msg is not None:
            line += msg
    return line