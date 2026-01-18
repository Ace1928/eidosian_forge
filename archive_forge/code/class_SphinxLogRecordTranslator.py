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
class SphinxLogRecordTranslator(logging.Filter):
    """Converts a log record to one Sphinx expects

    * Make a instance of SphinxLogRecord
    * docname to path if location given
    """
    LogRecordClass: Type[logging.LogRecord]

    def __init__(self, app: 'Sphinx') -> None:
        self.app = app
        super().__init__()

    def filter(self, record: SphinxWarningLogRecord) -> bool:
        if isinstance(record, logging.LogRecord):
            record.__class__ = self.LogRecordClass
        location = getattr(record, 'location', None)
        if isinstance(location, tuple):
            docname, lineno = location
            if docname and lineno:
                record.location = '%s:%s' % (self.app.env.doc2path(docname), lineno)
            elif docname:
                record.location = '%s' % self.app.env.doc2path(docname)
            else:
                record.location = None
        elif isinstance(location, nodes.Node):
            record.location = get_node_location(location)
        elif location and ':' not in location:
            record.location = '%s' % self.app.env.doc2path(location)
        return True