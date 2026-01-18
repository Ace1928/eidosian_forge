from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_hashbytes(args: t.List) -> exp.Expression:
    kind, data = args
    kind = kind.name.upper() if kind.is_string else ''
    if kind == 'MD5':
        args.pop(0)
        return exp.MD5(this=data)
    if kind in ('SHA', 'SHA1'):
        args.pop(0)
        return exp.SHA(this=data)
    if kind == 'SHA2_256':
        return exp.SHA2(this=data, length=exp.Literal.number(256))
    if kind == 'SHA2_512':
        return exp.SHA2(this=data, length=exp.Literal.number(512))
    return exp.func('HASHBYTES', *args)