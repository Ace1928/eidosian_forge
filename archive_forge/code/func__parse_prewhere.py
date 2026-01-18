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
def _parse_prewhere(self, skip_where_token: bool=False) -> t.Optional[exp.PreWhere]:
    if not skip_where_token and (not self._match(TokenType.PREWHERE)):
        return None
    return self.expression(exp.PreWhere, comments=self._prev_comments, this=self._parse_conjunction())