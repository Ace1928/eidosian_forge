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
def _flatten_excepthandler(node):
    if not isinstance(node, ast.Tuple):
        yield node
        return
    expr_list = node.elts.copy()
    while len(expr_list):
        expr = expr_list.pop(0)
        if isinstance(expr, ast.Starred) and isinstance(expr.value, (ast.List, ast.Tuple)):
            expr_list.extend(expr.value.elts)
            continue
        yield expr