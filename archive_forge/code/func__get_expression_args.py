from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
def _get_expression_args(expr: sym.Expr, params: dict[str, float]) -> list[np.ndarray | float]:
    """A helper function to get argument to evaluate expression.

    Args:
        expr: Symbolic expression to evaluate.
        params: Dictionary of parameter, which is a superset of expression arguments.

    Returns:
        Arguments passed to the lambdified expression.

    Raises:
        PulseError: When a free symbol value is not defined in the pulse instance parameters.
    """
    args: list[np.ndarray | float] = []
    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        if symbol.name == 't':
            times = np.arange(0, params['duration']) + 1 / 2
            args.insert(0, times)
            continue
        try:
            args.append(params[symbol.name])
        except KeyError as ex:
            raise PulseError(f"Pulse parameter '{symbol.name}' is not defined for this instance. Please check your waveform expression is correct.") from ex
    return args