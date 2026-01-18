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
def _parse_references(self, match: bool=True) -> t.Optional[exp.Reference]:
    if match and (not self._match(TokenType.REFERENCES)):
        return None
    expressions = None
    this = self._parse_table(schema=True)
    options = self._parse_key_constraint_options()
    return self.expression(exp.Reference, this=this, expressions=expressions, options=options)