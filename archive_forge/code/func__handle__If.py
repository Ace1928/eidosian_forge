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
def _handle__If(self, node: ast.If) -> set[ArcStart]:
    start = self.line_for_node(node.test)
    from_start = ArcStart(start, cause='the condition on line {lineno} was never true')
    exits = self.add_body_arcs(node.body, from_start=from_start)
    from_start = ArcStart(start, cause='the condition on line {lineno} was never false')
    exits |= self.add_body_arcs(node.orelse, from_start=from_start)
    return exits