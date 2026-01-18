from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
@staticmethod
def _is_infinite_iterator(node: ast.expr) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'itertools')):
        return False
    if node.func.attr in {'cycle', 'count'}:
        return True
    elif node.func.attr == 'repeat':
        if len(node.args) == 1 and len(node.keywords) == 0:
            return True
        if len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and (node.args[1].value is None):
            return True
        for kw in node.keywords:
            if kw.arg == 'times' and isinstance(kw.value, ast.Constant) and (kw.value.value is None):
                return True
    return False