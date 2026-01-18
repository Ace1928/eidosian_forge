from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_create(self) -> exp.Create | exp.Command:
    create = super()._parse_create()
    if isinstance(create, exp.Create):
        table = create.this.this if isinstance(create.this, exp.Schema) else create.this
        if isinstance(table, exp.Table) and table.this.args.get('temporary'):
            if not create.args.get('properties'):
                create.set('properties', exp.Properties(expressions=[]))
            create.args['properties'].append('expressions', exp.TemporaryProperty())
    return create