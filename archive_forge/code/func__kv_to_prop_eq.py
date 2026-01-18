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
def _kv_to_prop_eq(self, expressions: t.List[exp.Expression]) -> t.List[exp.Expression]:
    transformed = []
    for e in expressions:
        if isinstance(e, self.KEY_VALUE_DEFINITIONS):
            if isinstance(e, exp.Alias):
                e = self.expression(exp.PropertyEQ, this=e.args.get('alias'), expression=e.this)
            if not isinstance(e, exp.PropertyEQ):
                e = self.expression(exp.PropertyEQ, this=exp.to_identifier(e.this.name), expression=e.expression)
            if isinstance(e.this, exp.Column):
                e.this.replace(e.this.this)
        transformed.append(e)
    return transformed