from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from pandas.errors import NumExprClobberingError
from pandas.core.computation.align import (
from pandas.core.computation.ops import (
from pandas.io.formats import printing
def _check_ne_builtin_clash(expr: Expr) -> None:
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """
    names = expr.names
    overlap = names & _ne_builtins
    if overlap:
        s = ', '.join([repr(x) for x in overlap])
        raise NumExprClobberingError(f'Variables in expression "{expr}" overlap with builtins: ({s})')