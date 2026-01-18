from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _build_sort_array_desc(args: t.List) -> exp.Expression:
    return exp.SortArray(this=seq_get(args, 0), asc=exp.false())