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
def _parse_withisolatedloading(self) -> t.Optional[exp.IsolatedLoadingProperty]:
    index = self._index
    no = self._match_text_seq('NO')
    concurrent = self._match_text_seq('CONCURRENT')
    if not self._match_text_seq('ISOLATED', 'LOADING'):
        self._retreat(index)
        return None
    target = self._parse_var_from_options(self.ISOLATED_LOADING_OPTIONS, raise_unmatched=False)
    return self.expression(exp.IsolatedLoadingProperty, no=no, concurrent=concurrent, target=target)