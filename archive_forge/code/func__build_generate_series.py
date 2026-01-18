from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _build_generate_series(args: t.List) -> exp.GenerateSeries:
    step = seq_get(args, 2)
    if step is None:
        return exp.GenerateSeries.from_arg_list(args)
    if step.is_string:
        args[2] = exp.to_interval(step.this)
    elif isinstance(step, exp.Interval) and (not step.args.get('unit')):
        args[2] = exp.to_interval(step.this.this)
    return exp.GenerateSeries.from_arg_list(args)