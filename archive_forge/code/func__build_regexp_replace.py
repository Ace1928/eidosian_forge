from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_regexp_replace(args: t.List) -> exp.RegexpReplace:
    regexp_replace = exp.RegexpReplace.from_arg_list(args)
    if not regexp_replace.args.get('replacement'):
        regexp_replace.set('replacement', exp.Literal.string(''))
    return regexp_replace