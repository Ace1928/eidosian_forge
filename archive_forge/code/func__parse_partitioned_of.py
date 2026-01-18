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
def _parse_partitioned_of(self) -> t.Optional[exp.PartitionedOfProperty]:
    if not self._match_text_seq('OF'):
        self._retreat(self._index - 1)
        return None
    this = self._parse_table(schema=True)
    if self._match(TokenType.DEFAULT):
        expression: exp.Var | exp.PartitionBoundSpec = exp.var('DEFAULT')
    elif self._match_text_seq('FOR', 'VALUES'):
        expression = self._parse_partition_bound_spec()
    else:
        self.raise_error('Expecting either DEFAULT or FOR VALUES clause.')
    return self.expression(exp.PartitionedOfProperty, this=this, expression=expression)