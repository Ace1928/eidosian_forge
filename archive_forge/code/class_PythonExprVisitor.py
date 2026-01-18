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
@disallow(_unsupported_nodes | _python_not_supported | frozenset(['Not']))
class PythonExprVisitor(BaseExprVisitor):

    def __init__(self, env, engine, parser, preparser=lambda source, f=None: source) -> None:
        super().__init__(env, engine, parser, preparser=preparser)