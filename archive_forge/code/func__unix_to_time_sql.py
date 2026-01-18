from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _unix_to_time_sql(self: BigQuery.Generator, expression: exp.UnixToTime) -> str:
    scale = expression.args.get('scale')
    timestamp = expression.this
    if scale in (None, exp.UnixToTime.SECONDS):
        return self.func('TIMESTAMP_SECONDS', timestamp)
    if scale == exp.UnixToTime.MILLIS:
        return self.func('TIMESTAMP_MILLIS', timestamp)
    if scale == exp.UnixToTime.MICROS:
        return self.func('TIMESTAMP_MICROS', timestamp)
    unix_seconds = exp.cast(exp.Div(this=timestamp, expression=exp.func('POW', 10, scale)), exp.DataType.Type.BIGINT)
    return self.func('TIMESTAMP_SECONDS', unix_seconds)