from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _build_count_if(args: t.List) -> exp.CountIf | exp.CombinedAggFunc:
    if len(args) == 1:
        return exp.CountIf(this=seq_get(args, 0))
    return exp.CombinedAggFunc(this='countIf', expressions=args, parts=('count', 'If'))