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
def _is_assertRaises_like(node: ast.withitem) -> bool:
    if not (isinstance(node, ast.withitem) and isinstance(node.context_expr, ast.Call) and isinstance(node.context_expr.func, (ast.Attribute, ast.Name))):
        return False
    if isinstance(node.context_expr.func, ast.Name):
        return node.context_expr.func.id in B908_pytest_functions
    elif isinstance(node.context_expr.func, ast.Attribute) and isinstance(node.context_expr.func.value, ast.Name):
        return node.context_expr.func.value.id == 'pytest' and node.context_expr.func.attr in B908_pytest_functions or (node.context_expr.func.value.id == 'self' and node.context_expr.func.attr in B908_unittest_methods)
    else:
        return False