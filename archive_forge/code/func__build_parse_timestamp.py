from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _build_parse_timestamp(args: t.List) -> exp.StrToTime:
    this = build_formatted_time(exp.StrToTime, 'bigquery')([seq_get(args, 1), seq_get(args, 0)])
    this.set('zone', seq_get(args, 2))
    return this