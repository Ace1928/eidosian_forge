from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _build_from_unixtime(args: t.List) -> exp.Expression:
    if len(args) == 3:
        return exp.UnixToTime(this=seq_get(args, 0), hours=seq_get(args, 1), minutes=seq_get(args, 2))
    if len(args) == 2:
        return exp.UnixToTime(this=seq_get(args, 0), zone=seq_get(args, 1))
    return exp.UnixToTime.from_arg_list(args)