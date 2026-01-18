from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
class LoopBlock(Block):
    """A block on the block stack representing a `for` or `while` loop."""

    def __init__(self, start: TLineNo) -> None:
        self.start = start
        self.break_exits: set[ArcStart] = set()

    def process_break_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        self.break_exits.update(exits)
        return True

    def process_continue_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        for xit in exits:
            add_arc(xit.lineno, self.start, xit.cause)
        return True