from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_if(self) -> t.Optional[exp.Expression]:
    index = self._index
    if self._match_text_seq('OBJECT_ID'):
        self._parse_wrapped_csv(self._parse_string)
        if self._match_text_seq('IS', 'NOT', 'NULL') and self._match(TokenType.DROP):
            return self._parse_drop(exists=True)
        self._retreat(index)
    return super()._parse_if()