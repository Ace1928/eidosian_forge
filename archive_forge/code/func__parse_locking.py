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
def _parse_locking(self) -> exp.LockingProperty:
    if self._match(TokenType.TABLE):
        kind = 'TABLE'
    elif self._match(TokenType.VIEW):
        kind = 'VIEW'
    elif self._match(TokenType.ROW):
        kind = 'ROW'
    elif self._match_text_seq('DATABASE'):
        kind = 'DATABASE'
    else:
        kind = None
    if kind in ('DATABASE', 'TABLE', 'VIEW'):
        this = self._parse_table_parts()
    else:
        this = None
    if self._match(TokenType.FOR):
        for_or_in = 'FOR'
    elif self._match(TokenType.IN):
        for_or_in = 'IN'
    else:
        for_or_in = None
    if self._match_text_seq('ACCESS'):
        lock_type = 'ACCESS'
    elif self._match_texts(('EXCL', 'EXCLUSIVE')):
        lock_type = 'EXCLUSIVE'
    elif self._match_text_seq('SHARE'):
        lock_type = 'SHARE'
    elif self._match_text_seq('READ'):
        lock_type = 'READ'
    elif self._match_text_seq('WRITE'):
        lock_type = 'WRITE'
    elif self._match_text_seq('CHECKSUM'):
        lock_type = 'CHECKSUM'
    else:
        lock_type = None
    override = self._match_text_seq('OVERRIDE')
    return self.expression(exp.LockingProperty, this=this, kind=kind, for_or_in=for_or_in, lock_type=lock_type, override=override)