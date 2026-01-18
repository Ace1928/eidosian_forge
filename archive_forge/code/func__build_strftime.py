from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def _build_strftime(args: t.List) -> exp.Anonymous | exp.TimeToStr:
    if len(args) == 1:
        args.append(exp.CurrentTimestamp())
    if len(args) == 2:
        return exp.TimeToStr(this=exp.TsOrDsToTimestamp(this=args[1]), format=args[0])
    return exp.Anonymous(this='STRFTIME', expressions=args)