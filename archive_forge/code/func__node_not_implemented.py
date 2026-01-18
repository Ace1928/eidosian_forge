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
def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """

    def f(self, *args, **kwargs):
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return f