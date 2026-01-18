from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
def _delta_sql(self: Generator, expression: DATE_ADD_OR_DIFF) -> str:
    if cast and isinstance(expression, exp.TsOrDsAdd):
        expression = ts_or_ds_add_cast(expression)
    return self.func(name, unit_to_var(expression), expression.expression, expression.this)