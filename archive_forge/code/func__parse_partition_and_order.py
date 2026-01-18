from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_partition_and_order(self) -> t.Tuple[t.List[exp.Expression], t.Optional[exp.Expression]]:
    return (self._parse_csv(self._parse_conjunction) if self._match_set({TokenType.PARTITION_BY, TokenType.DISTRIBUTE_BY}) else [], super()._parse_order(skip_order_token=self._match(TokenType.SORT_BY)))