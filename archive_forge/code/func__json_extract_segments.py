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
def _json_extract_segments(self: Generator, expression: JSON_EXTRACT_TYPE) -> str:
    path = expression.expression
    if not isinstance(path, exp.JSONPath):
        return rename_func(name)(self, expression)
    segments = []
    for segment in path.expressions:
        path = self.sql(segment)
        if path:
            if isinstance(segment, exp.JSONPathPart) and (quoted_index or not isinstance(segment, exp.JSONPathSubscript)):
                path = f'{self.dialect.QUOTE_START}{path}{self.dialect.QUOTE_END}'
            segments.append(path)
    if op:
        return f' {op} '.join([self.sql(expression.this), *segments])
    return self.func(name, expression.this, *segments)