from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_selection(self, query: exp.Query, append: bool=True) -> exp.Query:
    if self._match(TokenType.L_BRACE):
        selects = self._parse_csv(self._parse_expression)
        if not self._match(TokenType.R_BRACE, expression=query):
            self.raise_error('Expecting }')
    else:
        expression = self._parse_expression()
        selects = [expression] if expression else []
    projections = {select.alias_or_name: select.this if isinstance(select, exp.Alias) else select for select in query.selects}
    selects = [select.transform(lambda s: (projections[s.name].copy() if s.name in projections else s) if isinstance(s, exp.Column) else s, copy=False) for select in selects]
    return query.select(*selects, append=append, copy=False)