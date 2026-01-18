from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _first_last_sql(self: Presto.Generator, expression: exp.Func) -> str:
    """
    Trino doesn't support FIRST / LAST as functions, but they're valid in the context
    of MATCH_RECOGNIZE, so we need to preserve them in that case. In all other cases
    they're converted into an ARBITRARY call.

    Reference: https://trino.io/docs/current/sql/match-recognize.html#logical-navigation-functions
    """
    if isinstance(expression.find_ancestor(exp.MatchRecognize, exp.Select), exp.MatchRecognize):
        return self.function_fallback_sql(expression)
    return rename_func('ARBITRARY')(self, expression)