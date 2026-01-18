from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
@dataclass(unsafe_hash=True)
class TernaryNode(BaseNode):
    condition: BaseNode
    questionmark: SymbolNode
    trueblock: BaseNode
    column: SymbolNode
    falseblock: BaseNode

    def __init__(self, condition: BaseNode, questionmark: SymbolNode, trueblock: BaseNode, column: SymbolNode, falseblock: BaseNode):
        super().__init__(condition.lineno, condition.colno, condition.filename)
        self.condition = condition
        self.questionmark = questionmark
        self.trueblock = trueblock
        self.column = column
        self.falseblock = falseblock