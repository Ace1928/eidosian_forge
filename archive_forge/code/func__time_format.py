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
def _time_format(self: Generator, expression: exp.UnixToStr | exp.StrToUnix) -> t.Optional[str]:
    """
        Returns the time format for a given expression, unless it's equivalent
        to the default time format of the dialect of interest.
        """
    time_format = self.format_time(expression)
    return time_format if time_format != Dialect.get_or_raise(dialect).TIME_FORMAT else None