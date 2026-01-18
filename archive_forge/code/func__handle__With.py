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
def _handle__With(self, node: ast.With) -> set[ArcStart]:
    start = self.line_for_node(node)
    if env.PYBEHAVIOR.exit_through_with:
        self.block_stack.append(WithBlock(start=start))
    exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
    if env.PYBEHAVIOR.exit_through_with:
        with_block = self.block_stack.pop()
        assert isinstance(with_block, WithBlock)
        with_exit = {ArcStart(start)}
        if exits:
            for xit in exits:
                self.add_arc(xit.lineno, start)
            exits = with_exit
        if with_block.break_from:
            self.process_break_exits(self._combine_finally_starts(with_block.break_from, with_exit))
        if with_block.continue_from:
            self.process_continue_exits(self._combine_finally_starts(with_block.continue_from, with_exit))
        if with_block.return_from:
            self.process_return_exits(self._combine_finally_starts(with_block.return_from, with_exit))
    return exits