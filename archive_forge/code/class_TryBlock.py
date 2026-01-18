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
class TryBlock(Block):
    """A block on the block stack representing a `try` block."""

    def __init__(self, handler_start: TLineNo | None, final_start: TLineNo | None) -> None:
        self.handler_start = handler_start
        self.final_start = final_start
        self.break_from: set[ArcStart] = set()
        self.continue_from: set[ArcStart] = set()
        self.raise_from: set[ArcStart] = set()
        self.return_from: set[ArcStart] = set()

    def process_break_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        if self.final_start is not None:
            self.break_from.update(exits)
            return True
        return False

    def process_continue_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        if self.final_start is not None:
            self.continue_from.update(exits)
            return True
        return False

    def process_raise_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        if self.handler_start is not None:
            for xit in exits:
                add_arc(xit.lineno, self.handler_start, xit.cause)
        else:
            assert self.final_start is not None
            self.raise_from.update(exits)
        return True

    def process_return_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        if self.final_start is not None:
            self.return_from.update(exits)
            return True
        return False