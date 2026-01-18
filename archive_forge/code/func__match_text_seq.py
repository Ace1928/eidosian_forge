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
def _match_text_seq(self, *texts, advance=True):
    index = self._index
    for text in texts:
        if self._curr and self._curr.text.upper() == text:
            self._advance()
        else:
            self._retreat(index)
            return None
    if not advance:
        self._retreat(index)
    return True