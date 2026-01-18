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
def _handle__Match(self, node: ast.Match) -> set[ArcStart]:
    start = self.line_for_node(node)
    last_start = start
    exits = set()
    had_wildcard = False
    for case in node.cases:
        case_start = self.line_for_node(case.pattern)
        pattern = case.pattern
        while isinstance(pattern, ast.MatchOr):
            pattern = pattern.patterns[-1]
        if isinstance(pattern, ast.MatchAs):
            had_wildcard = True
        self.add_arc(last_start, case_start, 'the pattern on line {lineno} always matched')
        from_start = ArcStart(case_start, cause='the pattern on line {lineno} never matched')
        exits |= self.add_body_arcs(case.body, from_start=from_start)
        last_start = case_start
    if not had_wildcard:
        exits.add(ArcStart(case_start, cause='the pattern on line {lineno} always matched'))
    return exits