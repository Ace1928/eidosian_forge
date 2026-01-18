from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _parse_table_parts(self, schema: bool=False, is_db_reference: bool=False, wildcard: bool=False) -> exp.Table:
    table = super()._parse_table_parts(schema=schema, is_db_reference=is_db_reference, wildcard=True)
    if not table.catalog:
        if table.db:
            parts = table.db.split('.')
            if len(parts) == 2 and (not table.args['db'].quoted):
                table.set('catalog', exp.Identifier(this=parts[0]))
                table.set('db', exp.Identifier(this=parts[1]))
        else:
            parts = table.name.split('.')
            if len(parts) == 2 and (not table.this.quoted):
                table.set('db', exp.Identifier(this=parts[0]))
                table.set('this', exp.Identifier(this=parts[1]))
    if any(('.' in p.name for p in table.parts)):
        catalog, db, this, *rest = (exp.to_identifier(p, quoted=True) for p in split_num_words('.'.join((p.name for p in table.parts)), '.', 3))
        if rest and this:
            this = exp.Dot.build([this, *rest])
        table = exp.Table(this=this, db=db, catalog=catalog, pivots=table.args.get('pivots'))
        table.meta['quoted_table'] = True
    return table