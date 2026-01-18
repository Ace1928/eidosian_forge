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
def _parse_alter_table_drop(self) -> t.List[exp.Expression]:
    index = self._index - 1
    partition_exists = self._parse_exists()
    if self._match(TokenType.PARTITION, advance=False):
        return self._parse_csv(lambda: self._parse_drop_partition(exists=partition_exists))
    self._retreat(index)
    return self._parse_csv(self._parse_drop_column)