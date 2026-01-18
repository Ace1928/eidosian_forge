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
def _get_ancestors(self, loader: Optional['BaseLoader']) -> List['_File']:
    ancestors = [self.file]
    for chunk in self.file.body.chunks:
        if isinstance(chunk, _ExtendsBlock):
            if not loader:
                raise ParseError('{% extends %} block found, but no template loader')
            template = loader.load(chunk.name, self.name)
            ancestors.extend(template._get_ancestors(loader))
    return ancestors