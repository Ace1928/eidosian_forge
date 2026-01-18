from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import rename_func, unit_to_var
from sqlglot.dialects.hive import _build_with_ignore_nulls
from sqlglot.dialects.spark2 import Spark2, temporary_storage_provider
from sqlglot.helper import ensure_list, seq_get
from sqlglot.transforms import (
def _build_datediff(args: t.List) -> exp.Expression:
    """
    Although Spark docs don't mention the "unit" argument, Spark3 added support for
    it at some point. Databricks also supports this variant (see below).

    For example, in spark-sql (v3.3.1):
    - SELECT DATEDIFF('2020-01-01', '2020-01-05') results in -4
    - SELECT DATEDIFF(day, '2020-01-01', '2020-01-05') results in 4

    See also:
    - https://docs.databricks.com/sql/language-manual/functions/datediff3.html
    - https://docs.databricks.com/sql/language-manual/functions/datediff.html
    """
    unit = None
    this = seq_get(args, 0)
    expression = seq_get(args, 1)
    if len(args) == 3:
        unit = this
        this = args[2]
    return exp.DateDiff(this=exp.TsOrDsToDate(this=this), expression=exp.TsOrDsToDate(this=expression), unit=unit)