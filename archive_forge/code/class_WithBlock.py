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
class WithBlock(Block):
    """A block on the block stack representing a `with` block."""

    def __init__(self, start: TLineNo) -> None:
        assert env.PYBEHAVIOR.exit_through_with
        self.start = start
        self.break_from: set[ArcStart] = set()
        self.continue_from: set[ArcStart] = set()
        self.return_from: set[ArcStart] = set()

    def _process_exits(self, exits: set[ArcStart], add_arc: TAddArcFn, from_set: set[ArcStart] | None=None) -> bool:
        """Helper to process the four kinds of exits."""
        for xit in exits:
            add_arc(xit.lineno, self.start, xit.cause)
        if from_set is not None:
            from_set.update(exits)
        return True

    def process_break_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        return self._process_exits(exits, add_arc, self.break_from)

    def process_continue_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        return self._process_exits(exits, add_arc, self.continue_from)

    def process_raise_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        return self._process_exits(exits, add_arc)

    def process_return_exits(self, exits: set[ArcStart], add_arc: TAddArcFn) -> bool:
        return self._process_exits(exits, add_arc, self.return_from)