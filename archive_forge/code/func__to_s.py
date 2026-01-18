from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def _to_s(node: t.Any, verbose: bool=False, level: int=0) -> str:
    """Generate a textual representation of an Expression tree"""
    indent = '\n' + '  ' * (level + 1)
    delim = f',{indent}'
    if isinstance(node, Expression):
        args = {k: v for k, v in node.args.items() if v is not None and v != [] or verbose}
        if (node.type or verbose) and (not isinstance(node, DataType)):
            args['_type'] = node.type
        if node.comments or verbose:
            args['_comments'] = node.comments
        if verbose:
            args['_id'] = id(node)
        if node.is_leaf():
            indent = ''
            delim = ', '
        items = delim.join([f'{k}={_to_s(v, verbose, level + 1)}' for k, v in args.items()])
        return f'{node.__class__.__name__}({indent}{items})'
    if isinstance(node, list):
        items = delim.join((_to_s(i, verbose, level + 1) for i in node))
        items = f'{indent}{items}' if items else ''
        return f'[{items}]'
    return indent.join(textwrap.dedent(str(node).strip('\n')).splitlines())