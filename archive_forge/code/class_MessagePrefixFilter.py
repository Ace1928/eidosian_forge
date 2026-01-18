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
class MessagePrefixFilter(logging.Filter):
    """Prepend prefix to all log records."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        if self.prefix:
            record.msg = self.prefix + ' ' + record.msg
        return True