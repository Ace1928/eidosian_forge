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
def _code_object__ClassDef(self, node: ast.ClassDef) -> None:
    start = self.line_for_node(node)
    self.add_arc(-start, start)
    exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
    for xit in exits:
        self.add_arc(xit.lineno, -start, xit.cause, f"didn't exit the body of class {node.name!r}")