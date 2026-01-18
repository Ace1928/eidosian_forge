from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
@dataclass(unsafe_hash=True)
class ElementaryNode(T.Generic[TV_TokenTypes], BaseNode):
    value: TV_TokenTypes
    bytespan: T.Tuple[int, int] = field(hash=False)

    def __init__(self, token: Token[TV_TokenTypes]):
        super().__init__(token.lineno, token.colno, token.filename)
        self.value = token.value
        self.bytespan = token.bytespan