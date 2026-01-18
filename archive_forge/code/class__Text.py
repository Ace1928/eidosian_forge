import datetime
from io import StringIO
import linecache
import os.path
import posixpath
import re
import threading
from tornado import escape
from tornado.log import app_log
from tornado.util import ObjectDict, exec_in, unicode_type
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO
import typing
class _Text(_Node):

    def __init__(self, value: str, line: int, whitespace: str) -> None:
        self.value = value
        self.line = line
        self.whitespace = whitespace

    def generate(self, writer: '_CodeWriter') -> None:
        value = self.value
        if '<pre>' not in value:
            value = filter_whitespace(self.whitespace, value)
        if value:
            writer.write_line('_tt_append(%r)' % escape.utf8(value), self.line)