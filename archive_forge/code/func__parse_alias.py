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
def _parse_alias(self, this: t.Optional[exp.Expression], explicit: bool=False) -> t.Optional[exp.Expression]:
    any_token = self._match(TokenType.ALIAS)
    comments = self._prev_comments
    if explicit and (not any_token):
        return this
    if self._match(TokenType.L_PAREN):
        aliases = self.expression(exp.Aliases, comments=comments, this=this, expressions=self._parse_csv(lambda: self._parse_id_var(any_token)))
        self._match_r_paren(aliases)
        return aliases
    alias = self._parse_id_var(any_token, tokens=self.ALIAS_TOKENS) or (self.STRING_ALIASES and self._parse_string_as_identifier())
    if alias:
        this = self.expression(exp.Alias, comments=comments, this=this, alias=alias)
        column = this.this
        if not this.comments and column and column.comments:
            this.comments = column.comments
            column.comments = None
    return this