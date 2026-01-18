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
def _runtest_for(self, item: nodes.Item, when: str) -> Generator[None, None, None]:
    """Implement the internals of the pytest_runtest_xxx() hooks."""
    with catching_logs(self.caplog_handler, level=self.log_level) as caplog_handler, catching_logs(self.report_handler, level=self.log_level) as report_handler:
        caplog_handler.reset()
        report_handler.reset()
        item.stash[caplog_records_key][when] = caplog_handler.records
        item.stash[caplog_handler_key] = caplog_handler
        try:
            yield
        finally:
            log = report_handler.stream.getvalue().strip()
            item.add_report_section(when, 'log', log)