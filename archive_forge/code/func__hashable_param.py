import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
def _hashable_param(param_tuples: AbstractSet[Tuple[Union[str, sympy.Expr], Union[value.Scalar, sympy.Expr]]], precision=10000000.0) -> FrozenSet[Tuple[str, Union[int, Tuple[int, int]]]]:
    """Hash circuit parameters using fixed precision.

    Circuit parameters can be complex but we also need to use them as
    dictionary keys. We secretly use these fixed-precision integers.
    """
    return frozenset(((k, _fix_precision(v, precision)) for k, v in param_tuples if isinstance(k, str)))