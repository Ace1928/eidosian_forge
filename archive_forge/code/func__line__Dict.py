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
def _line__Dict(self, node: ast.Dict) -> TLineNo:
    if node.keys:
        if node.keys[0] is not None:
            return node.keys[0].lineno
        else:
            return node.values[0].lineno
    else:
        return node.lineno