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
def _parse_opclass(self) -> t.Optional[exp.Expression]:
    this = self._parse_conjunction()
    if self._match_texts(self.OPCLASS_FOLLOW_KEYWORDS, advance=False):
        return this
    if not self._match_set(self.OPTYPE_FOLLOW_TOKENS, advance=False):
        return self.expression(exp.Opclass, this=this, expression=self._parse_table_parts())
    return this