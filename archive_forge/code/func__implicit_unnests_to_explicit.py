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
def _implicit_unnests_to_explicit(self, this: E) -> E:
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers as _norm
    refs = {_norm(this.args['from'].this.copy(), dialect=self.dialect).alias_or_name}
    for i, join in enumerate(this.args.get('joins') or []):
        table = join.this
        normalized_table = table.copy()
        normalized_table.meta['maybe_column'] = True
        normalized_table = _norm(normalized_table, dialect=self.dialect)
        if isinstance(table, exp.Table) and (not join.args.get('on')):
            if normalized_table.parts[0].name in refs:
                table_as_column = table.to_column()
                unnest = exp.Unnest(expressions=[table_as_column])
                if isinstance(table.args.get('alias'), exp.TableAlias):
                    table_as_column.replace(table_as_column.this)
                    exp.alias_(unnest, None, table=[table.args['alias'].this], copy=False)
                table.replace(unnest)
        refs.add(normalized_table.alias_or_name)
    return this