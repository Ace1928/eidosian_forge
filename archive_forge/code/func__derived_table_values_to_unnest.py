from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _derived_table_values_to_unnest(self: BigQuery.Generator, expression: exp.Values) -> str:
    if not expression.find_ancestor(exp.From, exp.Join):
        return self.values_sql(expression)
    structs = []
    alias = expression.args.get('alias')
    for tup in expression.find_all(exp.Tuple):
        field_aliases = alias.columns if alias and alias.columns else (f'_c{i}' for i in range(len(tup.expressions)))
        expressions = [exp.PropertyEQ(this=exp.to_identifier(name), expression=fld) for name, fld in zip(field_aliases, tup.expressions)]
        structs.append(exp.Struct(expressions=expressions))
    alias_name_only = exp.TableAlias(columns=[alias.this]) if alias else None
    return self.unnest_sql(exp.Unnest(expressions=[exp.array(*structs, copy=False)], alias=alias_name_only))