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
class IncludeTemplate(object):

    def __enter__(_) -> '_CodeWriter':
        return self

    def __exit__(_, *args: Any) -> None:
        self.current_template = self.include_stack.pop()[0]