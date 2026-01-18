import logging
import logging.handlers
from collections import defaultdict
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import get_source_line
from sphinx.errors import SphinxWarning
from sphinx.util.console import colorize
from sphinx.util.osutil import abspath
class SphinxLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter allowing ``type`` and ``subtype`` keywords."""
    KEYWORDS = ['type', 'subtype', 'location', 'nonl', 'color', 'once']

    def log(self, level: Union[int, str], msg: str, *args: Any, **kwargs: Any) -> None:
        if isinstance(level, int):
            super().log(level, msg, *args, **kwargs)
        else:
            levelno = LEVEL_NAMES[level]
            super().log(levelno, msg, *args, **kwargs)

    def verbose(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(VERBOSE, msg, *args, **kwargs)

    def process(self, msg: str, kwargs: Dict) -> Tuple[str, Dict]:
        extra = kwargs.setdefault('extra', {})
        for keyword in self.KEYWORDS:
            if keyword in kwargs:
                extra[keyword] = kwargs.pop(keyword)
        return (msg, kwargs)

    def handle(self, record: logging.LogRecord) -> None:
        self.logger.handle(record)