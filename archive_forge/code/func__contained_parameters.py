from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _contained_parameters(expression: ExpressionDesignator) -> Set[Parameter]:
    """
    Determine which parameters are contained in this expression.

    :param expression: expression involving parameters
    :return: set of parameters contained in this expression
    """
    if isinstance(expression, BinaryExp):
        return _contained_parameters(expression.op1) | _contained_parameters(expression.op2)
    elif isinstance(expression, Function):
        return _contained_parameters(expression.expression)
    elif isinstance(expression, Parameter):
        return {expression}
    else:
        return set()