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
def _parse_mergeblockratio(self, no: bool=False, default: bool=False) -> exp.MergeBlockRatioProperty:
    if self._match(TokenType.EQ):
        return self.expression(exp.MergeBlockRatioProperty, this=self._parse_number(), percent=self._match(TokenType.PERCENT))
    return self.expression(exp.MergeBlockRatioProperty, no=no, default=default)