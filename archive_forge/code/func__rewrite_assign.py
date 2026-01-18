from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    return (toknum, '==' if tokval == '=' else tokval)