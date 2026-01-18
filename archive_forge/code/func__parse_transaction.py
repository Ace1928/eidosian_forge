from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_transaction(self) -> exp.Transaction | exp.Command:
    """Applies to SQL Server and Azure SQL Database
            BEGIN { TRAN | TRANSACTION }
            [ { transaction_name | @tran_name_variable }
            [ WITH MARK [ 'description' ] ]
            ]
            """
    if self._match_texts(('TRAN', 'TRANSACTION')):
        transaction = self.expression(exp.Transaction, this=self._parse_id_var())
        if self._match_text_seq('WITH', 'MARK'):
            transaction.set('mark', self._parse_string())
        return transaction
    return self._parse_as_command(self._prev)