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
def _parse_match_against(self) -> exp.MatchAgainst:
    expressions = self._parse_csv(self._parse_column)
    self._match_text_seq(')', 'AGAINST', '(')
    this = self._parse_string()
    if self._match_text_seq('IN', 'NATURAL', 'LANGUAGE', 'MODE'):
        modifier = 'IN NATURAL LANGUAGE MODE'
        if self._match_text_seq('WITH', 'QUERY', 'EXPANSION'):
            modifier = f'{modifier} WITH QUERY EXPANSION'
    elif self._match_text_seq('IN', 'BOOLEAN', 'MODE'):
        modifier = 'IN BOOLEAN MODE'
    elif self._match_text_seq('WITH', 'QUERY', 'EXPANSION'):
        modifier = 'WITH QUERY EXPANSION'
    else:
        modifier = None
    return self.expression(exp.MatchAgainst, this=this, expressions=expressions, modifier=modifier)