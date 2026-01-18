from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _is_derived_table(expression: exp.Subquery) -> bool:
    """
    We represent (tbl1 JOIN tbl2) as a Subquery, but it's not really a "derived table",
    as it doesn't introduce a new scope. If an alias is present, it shadows all names
    under the Subquery, so that's one exception to this rule.
    """
    return isinstance(expression, exp.Subquery) and bool(expression.alias or isinstance(expression.this, exp.UNWRAPPED_QUERIES))