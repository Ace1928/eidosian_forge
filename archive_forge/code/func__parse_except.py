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
def _parse_except(self) -> t.Optional[t.List[exp.Expression]]:
    if not self._match(TokenType.EXCEPT):
        return None
    if self._match(TokenType.L_PAREN, advance=False):
        return self._parse_wrapped_csv(self._parse_column)
    except_column = self._parse_column()
    return [except_column] if except_column else None