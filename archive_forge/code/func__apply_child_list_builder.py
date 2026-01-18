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
def _apply_child_list_builder(*expressions, instance, arg, append=True, copy=True, prefix=None, into=None, dialect=None, properties=None, **opts):
    instance = maybe_copy(instance, copy)
    parsed = []
    for expression in expressions:
        if expression is not None:
            if _is_wrong_expression(expression, into):
                expression = into(expressions=[expression])
            expression = maybe_parse(expression, into=into, dialect=dialect, prefix=prefix, **opts)
            parsed.extend(expression.expressions)
    existing = instance.args.get(arg)
    if append and existing:
        parsed = existing.expressions + parsed
    child = into(expressions=parsed)
    for k, v in (properties or {}).items():
        child.set(k, v)
    instance.set(arg, child)
    return instance