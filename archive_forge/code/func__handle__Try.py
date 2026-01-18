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
def _handle__Try(self, node: ast.Try) -> set[ArcStart]:
    if node.handlers:
        handler_start = self.line_for_node(node.handlers[0])
    else:
        handler_start = None
    if node.finalbody:
        final_start = self.line_for_node(node.finalbody[0])
    else:
        final_start = None
    assert handler_start is not None or final_start is not None
    try_block = TryBlock(handler_start, final_start)
    self.block_stack.append(try_block)
    start = self.line_for_node(node)
    exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
    if node.finalbody:
        try_block.handler_start = None
        if node.handlers:
            try_block.raise_from = set()
    else:
        self.block_stack.pop()
    handler_exits: set[ArcStart] = set()
    if node.handlers:
        last_handler_start: TLineNo | None = None
        for handler_node in node.handlers:
            handler_start = self.line_for_node(handler_node)
            if last_handler_start is not None:
                self.add_arc(last_handler_start, handler_start)
            last_handler_start = handler_start
            from_cause = "the exception caught by line {lineno} didn't happen"
            from_start = ArcStart(handler_start, cause=from_cause)
            handler_exits |= self.add_body_arcs(handler_node.body, from_start=from_start)
    if node.orelse:
        exits = self.add_body_arcs(node.orelse, prev_starts=exits)
    exits |= handler_exits
    if node.finalbody:
        self.block_stack.pop()
        final_from = exits | try_block.break_from | try_block.continue_from | try_block.raise_from | try_block.return_from
        final_exits = self.add_body_arcs(node.finalbody, prev_starts=final_from)
        if try_block.break_from:
            if env.PYBEHAVIOR.finally_jumps_back:
                for break_line in try_block.break_from:
                    lineno = break_line.lineno
                    cause = break_line.cause.format(lineno=lineno)
                    for final_exit in final_exits:
                        self.add_arc(final_exit.lineno, lineno, cause)
                breaks = try_block.break_from
            else:
                breaks = self._combine_finally_starts(try_block.break_from, final_exits)
            self.process_break_exits(breaks)
        if try_block.continue_from:
            if env.PYBEHAVIOR.finally_jumps_back:
                for continue_line in try_block.continue_from:
                    lineno = continue_line.lineno
                    cause = continue_line.cause.format(lineno=lineno)
                    for final_exit in final_exits:
                        self.add_arc(final_exit.lineno, lineno, cause)
                continues = try_block.continue_from
            else:
                continues = self._combine_finally_starts(try_block.continue_from, final_exits)
            self.process_continue_exits(continues)
        if try_block.raise_from:
            self.process_raise_exits(self._combine_finally_starts(try_block.raise_from, final_exits))
        if try_block.return_from:
            if env.PYBEHAVIOR.finally_jumps_back:
                for return_line in try_block.return_from:
                    lineno = return_line.lineno
                    cause = return_line.cause.format(lineno=lineno)
                    for final_exit in final_exits:
                        self.add_arc(final_exit.lineno, lineno, cause)
                returns = try_block.return_from
            else:
                returns = self._combine_finally_starts(try_block.return_from, final_exits)
            self.process_return_exits(returns)
        if exits:
            exits = final_exits
    return exits