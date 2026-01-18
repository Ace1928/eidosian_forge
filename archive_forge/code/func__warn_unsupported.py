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
def _warn_unsupported(self) -> None:
    if len(self._tokens) <= 1:
        return
    sql = self._find_sql(self._tokens[0], self._tokens[-1])[:self.error_message_context]
    logger.warning(f"'{sql}' contains unsupported syntax. Falling back to parsing as a 'Command'.")