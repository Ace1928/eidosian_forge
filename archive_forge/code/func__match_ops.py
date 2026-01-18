from __future__ import annotations
import ast
import copy
import operator
import cmath
from collections.abc import Callable
from typing import Any
from qiskit.pulse.exceptions import PulseError
from qiskit.circuit import ParameterExpression
@staticmethod
def _match_ops(opr: ast.AST, opr_dict: dict, *args) -> complex:
    """Helper method to apply operators.

        Args:
            opr: Operator of node.
            opr_dict: Mapper from ast to operator.
            *args: Arguments supplied to operator.

        Returns:
            Evaluated value.

        Raises:
            PulseError: When unsupported operation is specified.
        """
    for op_type, op_func in opr_dict.items():
        if isinstance(opr, op_type):
            return op_func(*args)
    raise PulseError('Operator %s is not supported.' % opr.__class__.__name__)