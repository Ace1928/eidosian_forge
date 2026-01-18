from __future__ import annotations
import typing as t
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import dict_depth
from sqlglot.schema import AbstractMappingSchema, normalize_name
def _ensure_tables(d: t.Optional[t.Dict], dialect: DialectType=None) -> t.Dict:
    if not d:
        return {}
    depth = dict_depth(d)
    if depth > 1:
        return {normalize_name(k, dialect=dialect, is_table=True).name: _ensure_tables(v, dialect=dialect) for k, v in d.items()}
    result = {}
    for table_name, table in d.items():
        table_name = normalize_name(table_name, dialect=dialect).name
        if isinstance(table, Table):
            result[table_name] = table
        else:
            table = [{normalize_name(column_name, dialect=dialect).name: value for column_name, value in row.items()} for row in table]
            column_names = tuple((column_name for column_name in table[0])) if table else ()
            rows = [tuple((row[name] for name in column_names)) for row in table]
            result[table_name] = Table(columns=column_names, rows=rows)
    return result