import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
def _extract_anchors_from_expr(segment: str) -> Optional[_Anchors]:
    """
    Given source code `segment` corresponding to a bytecode
    instruction, determine:
        - for binary ops, the location of the binary op
        - for indexing, the location of the brackets.
    `segment` is expected to be a valid Python expression
    """
    assert sys.version_info >= (3, 11)
    import ast
    try:
        tree = ast.parse('(\n' + segment + '\n)')
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    lines = segment.split('\n')

    def normalize(lineno, offset):
        return _fix_offset(lines[lineno], offset)

    def next_valid_char(lineno, col):
        while lineno < len(lines) and col >= len(lines[lineno]):
            col = 0
            lineno += 1
        assert lineno < len(lines) and col < len(lines[lineno])
        return (lineno, col)

    def increment(lineno, col):
        col += 1
        lineno, col = next_valid_char(lineno, col)
        assert lineno < len(lines) and col < len(lines[lineno])
        return (lineno, col)

    def nextline(lineno, col):
        col = 0
        lineno += 1
        lineno, col = next_valid_char(lineno, col)
        assert lineno < len(lines) and col < len(lines[lineno])
        return (lineno, col)
    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expr = statement.value
        if isinstance(expr, ast.BinOp):
            cur_lineno = cast(int, expr.left.end_lineno) - 2
            cur_col = normalize(cur_lineno, expr.left.end_col_offset)
            cur_lineno, cur_col = next_valid_char(cur_lineno, cur_col)
            while (ch := lines[cur_lineno][cur_col]).isspace() or ch in ')\\#':
                if ch in '\\#':
                    cur_lineno, cur_col = nextline(cur_lineno, cur_col)
                else:
                    cur_lineno, cur_col = increment(cur_lineno, cur_col)
            right_col = cur_col + 1
            if right_col < len(lines[cur_lineno]) and (not (ch := lines[cur_lineno][right_col]).isspace()) and (ch not in '\\#'):
                right_col += 1
            return _Anchors(cur_lineno, cur_col, cur_lineno, right_col)
        elif isinstance(expr, ast.Subscript):
            left_lineno = cast(int, expr.value.end_lineno) - 2
            left_col = normalize(left_lineno, expr.value.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != '[':
                left_lineno, left_col = increment(left_lineno, left_col)
            right_lineno = cast(int, expr.end_lineno) - 2
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)
        elif isinstance(expr, ast.Call):
            left_lineno = cast(int, expr.func.end_lineno) - 2
            left_col = normalize(left_lineno, expr.func.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != '(':
                left_lineno, left_col = increment(left_lineno, left_col)
            right_lineno = cast(int, expr.end_lineno) - 2
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)
    return None