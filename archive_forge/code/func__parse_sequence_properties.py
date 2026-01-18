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
def _parse_sequence_properties(self) -> t.Optional[exp.SequenceProperties]:
    seq = exp.SequenceProperties()
    options = []
    index = self._index
    while self._curr:
        if self._match_text_seq('INCREMENT'):
            self._match_text_seq('BY')
            self._match_text_seq('=')
            seq.set('increment', self._parse_term())
        elif self._match_text_seq('MINVALUE'):
            seq.set('minvalue', self._parse_term())
        elif self._match_text_seq('MAXVALUE'):
            seq.set('maxvalue', self._parse_term())
        elif self._match(TokenType.START_WITH) or self._match_text_seq('START'):
            self._match_text_seq('=')
            seq.set('start', self._parse_term())
        elif self._match_text_seq('CACHE'):
            seq.set('cache', self._parse_number() or True)
        elif self._match_text_seq('OWNED', 'BY'):
            seq.set('owned', None if self._match_text_seq('NONE') else self._parse_column())
        else:
            opt = self._parse_var_from_options(self.CREATE_SEQUENCE, raise_unmatched=False)
            if opt:
                options.append(opt)
            else:
                break
    seq.set('options', options if options else None)
    return None if self._index == index else seq