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
def _parse_var_from_options(self, options: OPTIONS_TYPE, raise_unmatched: bool=True) -> t.Optional[exp.Var]:
    start = self._curr
    if not start:
        return None
    option = start.text.upper()
    continuations = options.get(option)
    index = self._index
    self._advance()
    for keywords in continuations or []:
        if isinstance(keywords, str):
            keywords = (keywords,)
        if self._match_text_seq(*keywords):
            option = f'{option} {' '.join(keywords)}'
            break
    else:
        if continuations or continuations is None:
            if raise_unmatched:
                self.raise_error(f'Unknown option {option}')
            self._retreat(index)
            return None
    return exp.var(option)