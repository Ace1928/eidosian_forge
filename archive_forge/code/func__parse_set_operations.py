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
def _parse_set_operations(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    while this and self._match_set(self.SET_OPERATIONS):
        token_type = self._prev.token_type
        if token_type == TokenType.UNION:
            operation = exp.Union
        elif token_type == TokenType.EXCEPT:
            operation = exp.Except
        else:
            operation = exp.Intersect
        comments = self._prev.comments
        distinct = self._match(TokenType.DISTINCT) or not self._match(TokenType.ALL)
        by_name = self._match_text_seq('BY', 'NAME')
        expression = self._parse_select(nested=True, parse_set_operation=False)
        this = self.expression(operation, comments=comments, this=this, distinct=distinct, by_name=by_name, expression=expression)
    if isinstance(this, exp.Union) and self.MODIFIERS_ATTACHED_TO_UNION:
        expression = this.expression
        if expression:
            for arg in self.UNION_MODIFIERS:
                expr = expression.args.get(arg)
                if expr:
                    this.set(arg, expr.pop())
    return this