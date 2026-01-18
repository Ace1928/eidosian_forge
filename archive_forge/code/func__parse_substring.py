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
def _parse_substring(self) -> exp.Substring:
    args = t.cast(t.List[t.Optional[exp.Expression]], self._parse_csv(self._parse_bitwise))
    if self._match(TokenType.FROM):
        args.append(self._parse_bitwise())
        if self._match(TokenType.FOR):
            args.append(self._parse_bitwise())
    return self.validate_expression(exp.Substring.from_arg_list(args), args)