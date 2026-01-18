from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _array_contains_sql(self: BigQuery.Generator, expression: exp.ArrayContains) -> str:
    return self.sql(exp.Exists(this=exp.select('1').from_(exp.Unnest(expressions=[expression.left]).as_('_unnest', table=['_col'])).where(exp.column('_col').eq(expression.right))))