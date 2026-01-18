from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_xml_table(self) -> exp.XMLTable:
    this = self._parse_string()
    passing = None
    columns = None
    if self._match_text_seq('PASSING'):
        self._match_text_seq('BY', 'VALUE')
        passing = self._parse_csv(self._parse_column)
    by_ref = self._match_text_seq('RETURNING', 'SEQUENCE', 'BY', 'REF')
    if self._match_text_seq('COLUMNS'):
        columns = self._parse_csv(self._parse_field_def)
    return self.expression(exp.XMLTable, this=this, passing=passing, columns=columns, by_ref=by_ref)