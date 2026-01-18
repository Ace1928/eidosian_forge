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
def _parse_column_def(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    if isinstance(this, exp.Column):
        this = this.this
    kind = self._parse_types(schema=True)
    if self._match_text_seq('FOR', 'ORDINALITY'):
        return self.expression(exp.ColumnDef, this=this, ordinality=True)
    constraints: t.List[exp.Expression] = []
    if not kind and self._match(TokenType.ALIAS) or self._match_texts(('ALIAS', 'MATERIALIZED')):
        persisted = self._prev.text.upper() == 'MATERIALIZED'
        constraints.append(self.expression(exp.ComputedColumnConstraint, this=self._parse_conjunction(), persisted=persisted or self._match_text_seq('PERSISTED'), not_null=self._match_pair(TokenType.NOT, TokenType.NULL)))
    elif kind and self._match_pair(TokenType.ALIAS, TokenType.L_PAREN, advance=False):
        self._match(TokenType.ALIAS)
        constraints.append(self.expression(exp.TransformColumnConstraint, this=self._parse_field()))
    while True:
        constraint = self._parse_column_constraint()
        if not constraint:
            break
        constraints.append(constraint)
    if not kind and (not constraints):
        return this
    return self.expression(exp.ColumnDef, this=this, kind=kind, constraints=constraints)