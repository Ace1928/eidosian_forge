from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _contained_mrefs(expression: ExpressionDesignator) -> Set[MemoryReference]:
    """
    Determine which memory references are contained in this expression.

    :param expression: expression involving parameters
    :return: set of parameters contained in this expression
    """
    if isinstance(expression, BinaryExp):
        return _contained_mrefs(expression.op1) | _contained_mrefs(expression.op2)
    elif isinstance(expression, Function):
        return _contained_mrefs(expression.expression)
    elif isinstance(expression, MemoryReference):
        return {expression}
    else:
        return set()