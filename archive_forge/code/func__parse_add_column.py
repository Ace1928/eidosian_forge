from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def _parse_add_column(self) -> t.Optional[exp.Expression]:
    return self._match_text_seq('ADD', 'COLUMNS') and self._parse_schema()