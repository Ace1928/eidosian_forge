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
def _parse_partition_bound_expr() -> t.Optional[exp.Expression]:
    if self._match_text_seq('MINVALUE'):
        return exp.var('MINVALUE')
    if self._match_text_seq('MAXVALUE'):
        return exp.var('MAXVALUE')
    return self._parse_bitwise()