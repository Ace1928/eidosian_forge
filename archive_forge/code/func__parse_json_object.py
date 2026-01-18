from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _parse_json_object(self, agg=False):
    json_object = super()._parse_json_object()
    array_kv_pair = seq_get(json_object.expressions, 0)
    if array_kv_pair and isinstance(array_kv_pair.this, exp.Array) and isinstance(array_kv_pair.expression, exp.Array):
        keys = array_kv_pair.this.expressions
        values = array_kv_pair.expression.expressions
        json_object.set('expressions', [exp.JSONKeyValue(this=k, expression=v) for k, v in zip(keys, values)])
    return json_object