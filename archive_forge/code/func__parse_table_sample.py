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
def _parse_table_sample(self, as_modifier: bool=False) -> t.Optional[exp.TableSample]:
    if not self._match(TokenType.TABLE_SAMPLE) and (not (as_modifier and self._match_text_seq('USING', 'SAMPLE'))):
        return None
    bucket_numerator = None
    bucket_denominator = None
    bucket_field = None
    percent = None
    size = None
    seed = None
    method = self._parse_var(tokens=(TokenType.ROW,), upper=True)
    matched_l_paren = self._match(TokenType.L_PAREN)
    if self.TABLESAMPLE_CSV:
        num = None
        expressions = self._parse_csv(self._parse_primary)
    else:
        expressions = None
        num = self._parse_factor() if self._match(TokenType.NUMBER, advance=False) else self._parse_primary() or self._parse_placeholder()
    if self._match_text_seq('BUCKET'):
        bucket_numerator = self._parse_number()
        self._match_text_seq('OUT', 'OF')
        bucket_denominator = bucket_denominator = self._parse_number()
        self._match(TokenType.ON)
        bucket_field = self._parse_field()
    elif self._match_set((TokenType.PERCENT, TokenType.MOD)):
        percent = num
    elif self._match(TokenType.ROWS) or not self.dialect.TABLESAMPLE_SIZE_IS_PERCENT:
        size = num
    else:
        percent = num
    if matched_l_paren:
        self._match_r_paren()
    if self._match(TokenType.L_PAREN):
        method = self._parse_var(upper=True)
        seed = self._match(TokenType.COMMA) and self._parse_number()
        self._match_r_paren()
    elif self._match_texts(('SEED', 'REPEATABLE')):
        seed = self._parse_wrapped(self._parse_number)
    return self.expression(exp.TableSample, expressions=expressions, method=method, bucket_numerator=bucket_numerator, bucket_denominator=bucket_denominator, bucket_field=bucket_field, percent=percent, size=size, seed=seed)