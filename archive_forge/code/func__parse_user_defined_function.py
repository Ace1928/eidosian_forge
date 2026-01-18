from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_user_defined_function(self, kind: t.Optional[TokenType]=None) -> t.Optional[exp.Expression]:
    this = super()._parse_user_defined_function(kind=kind)
    if kind == TokenType.FUNCTION or isinstance(this, exp.UserDefinedFunction) or self._match(TokenType.ALIAS, advance=False):
        return this
    expressions = self._parse_csv(self._parse_function_parameter)
    return self.expression(exp.UserDefinedFunction, this=this, expressions=expressions)