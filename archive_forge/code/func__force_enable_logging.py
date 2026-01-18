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
def _force_enable_logging(self, level: Union[int, str], logger_obj: logging.Logger) -> int:
    """Enable the desired logging level if the global level was disabled via ``logging.disabled``.

        Only enables logging levels greater than or equal to the requested ``level``.

        Does nothing if the desired ``level`` wasn't disabled.

        :param level:
            The logger level caplog should capture.
            All logging is enabled if a non-standard logging level string is supplied.
            Valid level strings are in :data:`logging._nameToLevel`.
        :param logger_obj: The logger object to check.

        :return: The original disabled logging level.
        """
    original_disable_level: int = logger_obj.manager.disable
    if isinstance(level, str):
        level = logging.getLevelName(level)
    if not isinstance(level, int):
        logging.disable(logging.NOTSET)
    elif not logger_obj.isEnabledFor(level):
        disable_level = max(level - 10, logging.NOTSET)
        logging.disable(disable_level)
    return original_disable_level