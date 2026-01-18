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
def _parse_pivot(self) -> t.Optional[exp.Pivot]:
    index = self._index
    include_nulls = None
    if self._match(TokenType.PIVOT):
        unpivot = False
    elif self._match(TokenType.UNPIVOT):
        unpivot = True
        if self._match_text_seq('INCLUDE', 'NULLS'):
            include_nulls = True
        elif self._match_text_seq('EXCLUDE', 'NULLS'):
            include_nulls = False
    else:
        return None
    expressions = []
    if not self._match(TokenType.L_PAREN):
        self._retreat(index)
        return None
    if unpivot:
        expressions = self._parse_csv(self._parse_column)
    else:
        expressions = self._parse_csv(lambda: self._parse_alias(self._parse_function()))
    if not expressions:
        self.raise_error("Failed to parse PIVOT's aggregation list")
    if not self._match(TokenType.FOR):
        self.raise_error('Expecting FOR')
    field = self._parse_pivot_in()
    self._match_r_paren()
    pivot = self.expression(exp.Pivot, expressions=expressions, field=field, unpivot=unpivot, include_nulls=include_nulls)
    if not self._match_set((TokenType.PIVOT, TokenType.UNPIVOT), advance=False):
        pivot.set('alias', self._parse_table_alias())
    if not unpivot:
        names = self._pivot_column_names(t.cast(t.List[exp.Expression], expressions))
        columns: t.List[exp.Expression] = []
        for fld in pivot.args['field'].expressions:
            field_name = fld.sql() if self.IDENTIFY_PIVOT_STRINGS else fld.alias_or_name
            for name in names:
                if self.PREFIXED_PIVOT_COLUMNS:
                    name = f'{name}_{field_name}' if name else field_name
                else:
                    name = f'{field_name}_{name}' if name else field_name
                columns.append(exp.to_identifier(name))
        pivot.set('columns', columns)
    return pivot