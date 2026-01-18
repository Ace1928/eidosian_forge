from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_quantile(self) -> exp.Quantile:
    this = self._parse_lambda()
    params = self._parse_func_params()
    if params:
        return self.expression(exp.Quantile, this=params[0], quantile=this)
    return self.expression(exp.Quantile, this=this, quantile=exp.Literal.number(0.5))