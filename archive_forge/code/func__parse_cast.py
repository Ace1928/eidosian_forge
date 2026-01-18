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
def _parse_cast(self, strict: bool, safe: t.Optional[bool]=None) -> exp.Expression:
    this = self._parse_conjunction()
    if not self._match(TokenType.ALIAS):
        if self._match(TokenType.COMMA):
            return self.expression(exp.CastToStrType, this=this, to=self._parse_string())
        self.raise_error('Expected AS after CAST')
    fmt = None
    to = self._parse_types()
    if self._match(TokenType.FORMAT):
        fmt_string = self._parse_string()
        fmt = self._parse_at_time_zone(fmt_string)
        if not to:
            to = exp.DataType.build(exp.DataType.Type.UNKNOWN)
        if to.this in exp.DataType.TEMPORAL_TYPES:
            this = self.expression(exp.StrToDate if to.this == exp.DataType.Type.DATE else exp.StrToTime, this=this, format=exp.Literal.string(format_time(fmt_string.this if fmt_string else '', self.dialect.FORMAT_MAPPING or self.dialect.TIME_MAPPING, self.dialect.FORMAT_TRIE or self.dialect.TIME_TRIE)))
            if isinstance(fmt, exp.AtTimeZone) and isinstance(this, exp.StrToTime):
                this.set('zone', fmt.args['zone'])
            return this
    elif not to:
        self.raise_error('Expected TYPE after CAST')
    elif isinstance(to, exp.Identifier):
        to = exp.DataType.build(to.name, udt=True)
    elif to.this == exp.DataType.Type.CHAR:
        if self._match(TokenType.CHARACTER_SET):
            to = self.expression(exp.CharacterSet, this=self._parse_var_or_string())
    return self.expression(exp.Cast if strict else exp.TryCast, this=this, to=to, format=fmt, safe=safe, action=self._parse_var_from_options(self.CAST_ACTIONS, raise_unmatched=False))