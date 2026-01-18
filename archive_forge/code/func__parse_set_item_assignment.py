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
def _parse_set_item_assignment(self, kind: t.Optional[str]=None) -> t.Optional[exp.Expression]:
    index = self._index
    if kind in ('GLOBAL', 'SESSION') and self._match_text_seq('TRANSACTION'):
        return self._parse_set_transaction(global_=kind == 'GLOBAL')
    left = self._parse_primary() or self._parse_id_var()
    assignment_delimiter = self._match_texts(('=', 'TO'))
    if not left or (self.SET_REQUIRES_ASSIGNMENT_DELIMITER and (not assignment_delimiter)):
        self._retreat(index)
        return None
    right = self._parse_statement() or self._parse_id_var()
    this = self.expression(exp.EQ, this=left, expression=right)
    return self.expression(exp.SetItem, this=this, kind=kind)