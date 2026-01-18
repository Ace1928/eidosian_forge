from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _build_with_ignore_nulls(exp_class: t.Type[exp.Expression]) -> t.Callable[[t.List[exp.Expression]], exp.Expression]:

    def _parse(args: t.List[exp.Expression]) -> exp.Expression:
        this = exp_class(this=seq_get(args, 0))
        if seq_get(args, 1) == exp.true():
            return exp.IgnoreNulls(this=this)
        return this
    return _parse