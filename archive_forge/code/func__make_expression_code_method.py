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
def _make_expression_code_method(noun: str) -> Callable[[AstArcAnalyzer, ast.AST], None]:
    """A function to make methods for expression-based callable _code_object__ methods."""

    def _code_object__expression_callable(self: AstArcAnalyzer, node: ast.AST) -> None:
        start = self.line_for_node(node)
        self.add_arc(-start, start, None, f"didn't run the {noun} on line {start}")
        self.add_arc(start, -start, None, f"didn't finish the {noun} on line {start}")
    return _code_object__expression_callable