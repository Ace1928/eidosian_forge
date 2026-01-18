from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import csv_reader, name_sequence
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema
def _qualify(table: exp.Table) -> None:
    if isinstance(table.this, exp.Identifier):
        if not table.args.get('db'):
            table.set('db', db)
        if not table.args.get('catalog') and table.args.get('db'):
            table.set('catalog', catalog)