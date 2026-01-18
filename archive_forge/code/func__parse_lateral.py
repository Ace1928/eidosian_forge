from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_lateral(self) -> t.Optional[exp.Lateral]:
    lateral = super()._parse_lateral()
    if not lateral:
        return lateral
    if isinstance(lateral.this, exp.Explode):
        table_alias = lateral.args.get('alias')
        columns = [exp.to_identifier(col) for col in self.FLATTEN_COLUMNS]
        if table_alias and (not table_alias.args.get('columns')):
            table_alias.set('columns', columns)
        elif not table_alias:
            exp.alias_(lateral, '_flattened', table=columns, copy=False)
    return lateral