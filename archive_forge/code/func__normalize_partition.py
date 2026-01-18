from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func, unit_to_var
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider
from sqlglot.helper import ensure_list, seq_get
from sqlglot.transforms import (
def _normalize_partition(e: exp.Expression) -> exp.Expression:
    """Normalize the expressions in PARTITION BY (<expression>, <expression>, ...)"""
    if isinstance(e, str):
        return exp.to_identifier(e)
    if isinstance(e, exp.Literal):
        return exp.to_identifier(e.name)
    return e