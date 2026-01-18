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
def _parse_unnamed_constraint(self, constraints: t.Optional[t.Collection[str]]=None) -> t.Optional[exp.Expression]:
    if self._match(TokenType.IDENTIFIER, advance=False) or not self._match_texts(constraints or self.CONSTRAINT_PARSERS):
        return None
    constraint = self._prev.text.upper()
    if constraint not in self.CONSTRAINT_PARSERS:
        self.raise_error(f'No parser found for schema constraint {constraint}.')
    return self.CONSTRAINT_PARSERS[constraint](self)