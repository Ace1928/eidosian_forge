import contextlib
import errno
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from logging import Filter
from typing import Any, ClassVar, Generator, List, Optional, TextIO, Type
from pip._vendor.rich.console import (
from pip._vendor.rich.highlighter import NullHighlighter
from pip._vendor.rich.logging import RichHandler
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._internal.utils._log import VERBOSE, getLogger
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import DEPRECATION_MSG_PREFIX
from pip._internal.utils.misc import ensure_dir
class RichPipStreamHandler(RichHandler):
    KEYWORDS: ClassVar[Optional[List[str]]] = []

    def __init__(self, stream: Optional[TextIO], no_color: bool) -> None:
        super().__init__(console=Console(file=stream, no_color=no_color, soft_wrap=True), show_time=False, show_level=False, show_path=False, highlighter=NullHighlighter())

    def emit(self, record: logging.LogRecord) -> None:
        style: Optional[Style] = None
        assert isinstance(record.args, tuple)
        if getattr(record, 'rich', False):
            rich_renderable, = record.args
            assert isinstance(rich_renderable, (ConsoleRenderable, RichCast, str)), f'{rich_renderable} is not rich-console-renderable'
            renderable: RenderableType = IndentedRenderable(rich_renderable, indent=get_indentation())
        else:
            message = self.format(record)
            renderable = self.render_message(record, message)
            if record.levelno is not None:
                if record.levelno >= logging.ERROR:
                    style = Style(color='red')
                elif record.levelno >= logging.WARNING:
                    style = Style(color='yellow')
        try:
            self.console.print(renderable, overflow='ignore', crop=False, style=style)
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Called when logging is unable to log some output."""
        exc_class, exc = sys.exc_info()[:2]
        if exc_class and exc and (self.console.file is sys.stdout) and _is_broken_pipe_error(exc_class, exc):
            raise BrokenStdoutLoggingError()
        return super().handleError(record)