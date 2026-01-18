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
def _parse_connect(self, skip_start_token: bool=False) -> t.Optional[exp.Connect]:
    if skip_start_token:
        start = None
    elif self._match(TokenType.START_WITH):
        start = self._parse_conjunction()
    else:
        return None
    self._match(TokenType.CONNECT_BY)
    nocycle = self._match_text_seq('NOCYCLE')
    self.NO_PAREN_FUNCTION_PARSERS['PRIOR'] = lambda self: self.expression(exp.Prior, this=self._parse_bitwise())
    connect = self._parse_conjunction()
    self.NO_PAREN_FUNCTION_PARSERS.pop('PRIOR')
    if not start and self._match(TokenType.START_WITH):
        start = self._parse_conjunction()
    return self.expression(exp.Connect, start=start, connect=connect, nocycle=nocycle)