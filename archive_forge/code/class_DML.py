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
class DML(Expression):

    def returning(self, expression: ExpOrStr, dialect: DialectType=None, copy: bool=True, **opts) -> DML:
        """
        Set the RETURNING expression. Not supported by all dialects.

        Example:
            >>> delete("tbl").returning("*", dialect="postgres").sql()
            'DELETE FROM tbl RETURNING *'

        Args:
            expression: the SQL code strings to parse.
                If an `Expression` instance is passed, it will be used as-is.
            dialect: the dialect used to parse the input expressions.
            copy: if `False`, modify this expression instance in-place.
            opts: other options to use to parse the input expressions.

        Returns:
            Delete: the modified expression.
        """
        return _apply_builder(expression=expression, instance=self, arg='returning', prefix='RETURNING', dialect=dialect, copy=copy, into=Returning, **opts)