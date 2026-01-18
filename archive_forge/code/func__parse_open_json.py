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
def _parse_open_json(self) -> exp.OpenJSON:
    this = self._parse_bitwise()
    path = self._match(TokenType.COMMA) and self._parse_string()

    def _parse_open_json_column_def() -> exp.OpenJSONColumnDef:
        this = self._parse_field(any_token=True)
        kind = self._parse_types()
        path = self._parse_string()
        as_json = self._match_pair(TokenType.ALIAS, TokenType.JSON)
        return self.expression(exp.OpenJSONColumnDef, this=this, kind=kind, path=path, as_json=as_json)
    expressions = None
    if self._match_pair(TokenType.R_PAREN, TokenType.WITH):
        self._match_l_paren()
        expressions = self._parse_csv(_parse_open_json_column_def)
    return self.expression(exp.OpenJSON, this=this, path=path, expressions=expressions)