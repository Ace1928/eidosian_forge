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
def _apply_builder(expression, instance, arg, copy=True, prefix=None, into=None, dialect=None, into_arg='this', **opts):
    if _is_wrong_expression(expression, into):
        expression = into(**{into_arg: expression})
    instance = maybe_copy(instance, copy)
    expression = maybe_parse(sql_or_expression=expression, prefix=prefix, into=into, dialect=dialect, **opts)
    instance.set(arg, expression)
    return instance