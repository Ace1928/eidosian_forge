from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_formatted_time(exp_class: t.Type[E], full_format_mapping: t.Optional[bool]=None) -> t.Callable[[t.List], E]:

    def _builder(args: t.List) -> E:
        assert len(args) == 2
        return exp_class(this=exp.cast(args[1], exp.DataType.Type.DATETIME), format=exp.Literal.string(format_time(args[0].name.lower(), {**TSQL.TIME_MAPPING, **FULL_FORMAT_TIME_MAPPING} if full_format_mapping else TSQL.TIME_MAPPING)))
    return _builder