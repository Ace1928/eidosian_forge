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
def _parse_blockcompression(self) -> exp.BlockCompressionProperty:
    self._match(TokenType.EQ)
    always = self._match_text_seq('ALWAYS')
    manual = self._match_text_seq('MANUAL')
    never = self._match_text_seq('NEVER')
    default = self._match_text_seq('DEFAULT')
    autotemp = None
    if self._match_text_seq('AUTOTEMP'):
        autotemp = self._parse_schema()
    return self.expression(exp.BlockCompressionProperty, always=always, manual=manual, never=never, default=default, autotemp=autotemp)