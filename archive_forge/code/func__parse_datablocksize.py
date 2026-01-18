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
def _parse_datablocksize(self, default: t.Optional[bool]=None, minimum: t.Optional[bool]=None, maximum: t.Optional[bool]=None) -> exp.DataBlocksizeProperty:
    self._match(TokenType.EQ)
    size = self._parse_number()
    units = None
    if self._match_texts(('BYTES', 'KBYTES', 'KILOBYTES')):
        units = self._prev.text
    return self.expression(exp.DataBlocksizeProperty, size=size, units=units, default=default, minimum=minimum, maximum=maximum)