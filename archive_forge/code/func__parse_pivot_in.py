from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_pivot_in(self) -> exp.In:

    def _parse_aliased_expression() -> t.Optional[exp.Expression]:
        this = self._parse_conjunction()
        self._match(TokenType.ALIAS)
        alias = self._parse_field()
        if alias:
            return self.expression(exp.PivotAlias, this=this, alias=alias)
        return this
    value = self._parse_column()
    if not self._match_pair(TokenType.IN, TokenType.L_PAREN):
        self.raise_error('Expecting IN (')
    aliased_expressions = self._parse_csv(_parse_aliased_expression)
    self._match_r_paren()
    return self.expression(exp.In, this=value, expressions=aliased_expressions)