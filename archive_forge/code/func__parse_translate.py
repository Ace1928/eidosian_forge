from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def _parse_translate(self, strict: bool) -> exp.Expression:
    this = self._parse_conjunction()
    if not self._match(TokenType.USING):
        self.raise_error('Expected USING in TRANSLATE')
    if self._match_texts(self.CHARSET_TRANSLATORS):
        charset_split = self._prev.text.split('_TO_')
        to = self.expression(exp.CharacterSet, this=charset_split[1])
    else:
        self.raise_error('Expected a character set translator after USING in TRANSLATE')
    return self.expression(exp.Cast if strict else exp.TryCast, this=this, to=to)