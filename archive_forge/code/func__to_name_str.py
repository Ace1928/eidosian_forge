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
def _to_name_str(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _to_name_str(node.func)
    elif isinstance(node, ast.Attribute):
        inner = _to_name_str(node.value)
        if inner is None:
            return None
        return f'{inner}.{node.attr}'
    else:
        return None