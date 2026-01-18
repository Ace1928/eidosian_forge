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
def _line__Module(self, node: ast.Module) -> TLineNo:
    if env.PYBEHAVIOR.module_firstline_1:
        return 1
    elif node.body:
        return self.line_for_node(node.body[0])
    else:
        return 1