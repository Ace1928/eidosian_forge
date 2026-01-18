from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_with_arg_as_text(klass: t.Type[exp.Expression]) -> t.Callable[[t.List[exp.Expression]], exp.Expression]:

    def _parse(args: t.List[exp.Expression]) -> exp.Expression:
        this = seq_get(args, 0)
        if this and (not this.is_string):
            this = exp.cast(this, exp.DataType.Type.TEXT)
        expression = seq_get(args, 1)
        kwargs = {'this': this}
        if expression:
            kwargs['expression'] = expression
        return klass(**kwargs)
    return _parse