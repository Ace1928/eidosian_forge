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
class _NamedBlock(_Node):

    def __init__(self, name: str, body: _Node, template: Template, line: int) -> None:
        self.name = name
        self.body = body
        self.template = template
        self.line = line

    def each_child(self) -> Iterable['_Node']:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        block = writer.named_blocks[self.name]
        with writer.include(block.template, self.line):
            block.body.generate(writer)

    def find_named_blocks(self, loader: Optional[BaseLoader], named_blocks: Dict[str, '_NamedBlock']) -> None:
        named_blocks[self.name] = self
        _Node.find_named_blocks(self, loader, named_blocks)