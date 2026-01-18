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
class WarningSuppressor(logging.Filter):
    """Filter logs by `suppress_warnings`."""

    def __init__(self, app: 'Sphinx') -> None:
        self.app = app
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        type = getattr(record, 'type', '')
        subtype = getattr(record, 'subtype', '')
        try:
            suppress_warnings = self.app.config.suppress_warnings
        except AttributeError:
            suppress_warnings = []
        if is_suppressed_warning(type, subtype, suppress_warnings):
            return False
        else:
            self.app._warncount += 1
            return True