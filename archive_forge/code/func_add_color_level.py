from contextlib import contextmanager
from contextlib import nullcontext
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import io
from io import StringIO
import logging
from logging import LogRecord
import os
from pathlib import Path
import re
from types import TracebackType
from typing import AbstractSet
from typing import Dict
from typing import final
from typing import Generator
from typing import Generic
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.capture import CaptureManager
from _pytest.config import _strtobool
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
def add_color_level(self, level: int, *color_opts: str) -> None:
    """Add or update color opts for a log level.

        :param level:
            Log level to apply a style to, e.g. ``logging.INFO``.
        :param color_opts:
            ANSI escape sequence color options. Capitalized colors indicates
            background color, i.e. ``'green', 'Yellow', 'bold'`` will give bold
            green text on yellow background.

        .. warning::
            This is an experimental API.
        """
    assert self._fmt is not None
    levelname_fmt_match = self.LEVELNAME_FMT_REGEX.search(self._fmt)
    if not levelname_fmt_match:
        return
    levelname_fmt = levelname_fmt_match.group()
    formatted_levelname = levelname_fmt % {'levelname': logging.getLevelName(level)}
    color_kwargs = {name: True for name in color_opts}
    colorized_formatted_levelname = self._terminalwriter.markup(formatted_levelname, **color_kwargs)
    self._level_to_fmt_mapping[level] = self.LEVELNAME_FMT_REGEX.sub(colorized_formatted_levelname, self._fmt)