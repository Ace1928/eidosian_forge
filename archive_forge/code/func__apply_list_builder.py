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
def _apply_list_builder(*expressions, instance, arg, append=True, copy=True, prefix=None, into=None, dialect=None, **opts):
    inst = maybe_copy(instance, copy)
    expressions = [maybe_parse(sql_or_expression=expression, into=into, prefix=prefix, dialect=dialect, **opts) for expression in expressions if expression is not None]
    existing_expressions = inst.args.get(arg)
    if append and existing_expressions:
        expressions = existing_expressions + expressions
    inst.set(arg, expressions)
    return inst