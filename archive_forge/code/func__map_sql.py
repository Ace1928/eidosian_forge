from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def _map_sql(self: Spark2.Generator, expression: exp.Map) -> str:
    keys = expression.args.get('keys')
    values = expression.args.get('values')
    if not keys or not values:
        return self.func('MAP')
    return self.func('MAP_FROM_ARRAYS', keys, values)