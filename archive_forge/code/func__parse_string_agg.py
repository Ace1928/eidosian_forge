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
def _parse_string_agg(self) -> exp.Expression:
    if self._match(TokenType.DISTINCT):
        args: t.List[t.Optional[exp.Expression]] = [self.expression(exp.Distinct, expressions=[self._parse_conjunction()])]
        if self._match(TokenType.COMMA):
            args.extend(self._parse_csv(self._parse_conjunction))
    else:
        args = self._parse_csv(self._parse_conjunction)
    index = self._index
    if not self._match(TokenType.R_PAREN) and args:
        args[-1] = self._parse_limit(this=self._parse_order(this=args[-1]))
        return self.expression(exp.GroupConcat, this=args[0], separator=seq_get(args, 1))
    if not self._match_text_seq('WITHIN', 'GROUP'):
        self._retreat(index)
        return self.validate_expression(exp.GroupConcat.from_arg_list(args), args)
    self._match_l_paren()
    order = self._parse_order(this=seq_get(args, 0))
    return self.expression(exp.GroupConcat, this=order, separator=seq_get(args, 1))