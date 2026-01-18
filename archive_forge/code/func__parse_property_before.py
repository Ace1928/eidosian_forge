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
def _parse_property_before(self) -> t.Optional[exp.Expression]:
    self._match(TokenType.COMMA)
    kwargs = {'no': self._match_text_seq('NO'), 'dual': self._match_text_seq('DUAL'), 'before': self._match_text_seq('BEFORE'), 'default': self._match_text_seq('DEFAULT'), 'local': self._match_text_seq('LOCAL') and 'LOCAL' or (self._match_text_seq('NOT', 'LOCAL') and 'NOT LOCAL'), 'after': self._match_text_seq('AFTER'), 'minimum': self._match_texts(('MIN', 'MINIMUM')), 'maximum': self._match_texts(('MAX', 'MAXIMUM'))}
    if self._match_texts(self.PROPERTY_PARSERS):
        parser = self.PROPERTY_PARSERS[self._prev.text.upper()]
        try:
            return parser(self, **{k: v for k, v in kwargs.items() if v})
        except TypeError:
            self.raise_error(f"Cannot parse property '{self._prev.text}'")
    return None