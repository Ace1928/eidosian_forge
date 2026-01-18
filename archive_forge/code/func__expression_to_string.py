from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _expression_to_string(expression: ExpressionDesignator) -> str:
    """
    Recursively converts an expression to a string taking into account precedence and associativity
    for placing parenthesis.

    :param expression: expression involving parameters
    :return: string such as '%x*(%y-4)'
    """
    if isinstance(expression, BinaryExp):
        left = _expression_to_string(expression.op1)
        if isinstance(expression.op1, BinaryExp) and (not (expression.op1.precedence > expression.precedence or (expression.op1.precedence == expression.precedence and expression.associates in ('left', 'both')))):
            left = '(' + left + ')'
        right = _expression_to_string(expression.op2)
        if isinstance(expression.op2, BinaryExp) and (not (expression.precedence < expression.op2.precedence or (expression.precedence == expression.op2.precedence and expression.associates in ('right', 'both')))):
            right = '(' + right + ')'
        elif isinstance(expression.op2, float) and ('pi' in right and right != 'pi'):
            right = '(' + right + ')'
        return left + expression.operator + right
    elif isinstance(expression, Function):
        return expression.name + '(' + _expression_to_string(expression.expression) + ')'
    elif isinstance(expression, Parameter):
        return str(expression)
    else:
        return format_parameter(expression)