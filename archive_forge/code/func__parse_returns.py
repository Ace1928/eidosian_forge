from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_returns(self) -> exp.ReturnsProperty:
    table = self._parse_id_var(any_token=False, tokens=self.RETURNS_TABLE_TOKENS)
    returns = super()._parse_returns()
    returns.set('table', table)
    return returns