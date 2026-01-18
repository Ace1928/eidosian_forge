from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def _raw_accept(self, node: mparser.BaseNode, data: T.Dict[str, T.Any]) -> None:
    old = self.current
    self.current = data
    node.accept(self)
    self.current = old