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
class _Module(_Expression):

    def __init__(self, expression: str, line: int) -> None:
        super().__init__('_tt_modules.' + expression, line, raw=True)