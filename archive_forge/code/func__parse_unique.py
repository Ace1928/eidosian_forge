from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_unique(self) -> exp.UniqueColumnConstraint:
    if self._match_texts(('CLUSTERED', 'NONCLUSTERED')):
        this = self.CONSTRAINT_PARSERS[self._prev.text.upper()](self)
    else:
        this = self._parse_schema(self._parse_id_var(any_token=False))
    return self.expression(exp.UniqueColumnConstraint, this=this)