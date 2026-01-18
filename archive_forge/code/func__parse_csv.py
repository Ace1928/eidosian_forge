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
def _parse_csv(self, parse_method: t.Callable, sep: TokenType=TokenType.COMMA) -> t.List[exp.Expression]:
    parse_result = parse_method()
    items = [parse_result] if parse_result is not None else []
    while self._match(sep):
        self._add_comments(parse_result)
        parse_result = parse_method()
        if parse_result is not None:
            items.append(parse_result)
    return items