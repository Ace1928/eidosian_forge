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
def _replace_placeholders(node: Expression, args, **kwargs) -> Expression:
    if isinstance(node, Placeholder):
        if node.this:
            new_name = kwargs.get(node.this)
            if new_name is not None:
                return convert(new_name)
        else:
            try:
                return convert(next(args))
            except StopIteration:
                pass
    return node