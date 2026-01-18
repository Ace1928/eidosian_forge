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
def _parse_withdata(self, no: bool=False) -> exp.WithDataProperty:
    if self._match_text_seq('AND', 'STATISTICS'):
        statistics = True
    elif self._match_text_seq('AND', 'NO', 'STATISTICS'):
        statistics = False
    else:
        statistics = None
    return self.expression(exp.WithDataProperty, no=no, statistics=statistics)