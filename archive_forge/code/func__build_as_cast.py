from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def _build_as_cast(to_type: str) -> t.Callable[[t.List], exp.Expression]:
    return lambda args: exp.Cast(this=seq_get(args, 0), to=exp.DataType.build(to_type))