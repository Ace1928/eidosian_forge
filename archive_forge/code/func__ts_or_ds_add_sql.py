from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _ts_or_ds_add_sql(self: BigQuery.Generator, expression: exp.TsOrDsAdd) -> str:
    return date_add_interval_sql('DATE', 'ADD')(self, ts_or_ds_add_cast(expression))