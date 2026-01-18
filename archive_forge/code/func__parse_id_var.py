from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_id_var(self, any_token: bool=True, tokens: t.Optional[t.Collection[TokenType]]=None) -> t.Optional[exp.Expression]:
    is_temporary = self._match(TokenType.HASH)
    is_global = is_temporary and self._match(TokenType.HASH)
    this = super()._parse_id_var(any_token=any_token, tokens=tokens)
    if this:
        if is_global:
            this.set('global', True)
        elif is_temporary:
            this.set('temporary', True)
    return this