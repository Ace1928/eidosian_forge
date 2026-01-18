from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_join(self, skip_join_token: bool=False, parse_bracket: bool=False) -> t.Optional[exp.Join]:
    join = super()._parse_join(skip_join_token=skip_join_token, parse_bracket=True)
    if join:
        join.set('global', join.args.pop('method', None))
    return join