from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
@dataclass(unsafe_hash=True)
class ParenthesizedNode(BaseNode):
    lpar: SymbolNode = field(hash=False)
    inner: BaseNode
    rpar: SymbolNode = field(hash=False)

    def __init__(self, lpar: SymbolNode, inner: BaseNode, rpar: SymbolNode):
        super().__init__(lpar.lineno, lpar.colno, inner.filename, end_lineno=rpar.lineno, end_colno=rpar.colno + 1)
        self.lpar = lpar
        self.inner = inner
        self.rpar = rpar