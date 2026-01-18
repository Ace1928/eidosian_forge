from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_decode(self) -> t.Optional[exp.Decode | exp.Case]:
    """
        There are generally two variants of the DECODE function:

        - DECODE(bin, charset)
        - DECODE(expression, search, result [, search, result] ... [, default])

        The second variant will always be parsed into a CASE expression. Note that NULL
        needs special treatment, since we need to explicitly check for it with `IS NULL`,
        instead of relying on pattern matching.
        """
    args = self._parse_csv(self._parse_conjunction)
    if len(args) < 3:
        return self.expression(exp.Decode, this=seq_get(args, 0), charset=seq_get(args, 1))
    expression, *expressions = args
    if not expression:
        return None
    ifs = []
    for search, result in zip(expressions[::2], expressions[1::2]):
        if not search or not result:
            return None
        if isinstance(search, exp.Literal):
            ifs.append(exp.If(this=exp.EQ(this=expression.copy(), expression=search), true=result))
        elif isinstance(search, exp.Null):
            ifs.append(exp.If(this=exp.Is(this=expression.copy(), expression=exp.Null()), true=result))
        else:
            cond = exp.or_(exp.EQ(this=expression.copy(), expression=search), exp.and_(exp.Is(this=expression.copy(), expression=exp.Null()), exp.Is(this=search.copy(), expression=exp.Null()), copy=False), copy=False)
            ifs.append(exp.If(this=cond, true=result))
    return exp.Case(ifs=ifs, default=expressions[-1] if len(expressions) % 2 == 1 else None)