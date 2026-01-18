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
def _parse_alter_table_add(self) -> t.List[exp.Expression]:
    index = self._index - 1
    if self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False):
        return self._parse_csv(lambda: self.expression(exp.AddConstraint, expressions=self._parse_csv(self._parse_constraint)))
    self._retreat(index)
    if not self.ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN and self._match_text_seq('ADD'):
        return self._parse_wrapped_csv(self._parse_field_def, optional=True)
    return self._parse_wrapped_csv(self._parse_add_column, optional=True)