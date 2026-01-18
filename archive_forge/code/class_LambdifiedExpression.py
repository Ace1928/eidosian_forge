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
class LambdifiedExpression:
    """Descriptor to lambdify symbolic expression with cache.

    When a new symbolic expression is assigned for the first time, :class:`.LambdifiedExpression`
    will internally lambdify the expressions and store the resulting callbacks in its cache.
    The next time it encounters the same expression it will return the cached callbacks
    thereby increasing the code's speed.

    Note that this class is a python `Descriptor`_, and thus not intended to be
    directly called by end-users. This class is designed to be attached to the
    :class:`.SymbolicPulse` as attributes for symbolic expressions.

    _`Descriptor`: https://docs.python.org/3/reference/datamodel.html#descriptors
    """

    def __init__(self, attribute: str):
        """Create new descriptor.

        Args:
            attribute: Name of attribute of :class:`.SymbolicPulse` that returns
                the target expression to evaluate.
        """
        self.attribute = attribute
        self.lambda_funcs: dict[int, Callable] = {}

    def __get__(self, instance, owner) -> Callable:
        expr = getattr(instance, self.attribute, None)
        if expr is None:
            raise PulseError(f"'{self.attribute}' of '{instance.pulse_type}' is not assigned.")
        key = hash(expr)
        if key not in self.lambda_funcs:
            self.__set__(instance, expr)
        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(value)
        if key not in self.lambda_funcs:
            params: list[Any] = []
            for p in sorted(value.free_symbols, key=lambda s: s.name):
                if p.name == 't':
                    params.insert(0, p)
                    continue
                params.append(p)
            try:
                lamb = sym.lambdify(params, [value], real=False)

                def _wrapped_lamb(*args):
                    if isinstance(args[0], np.ndarray):
                        t = args[0]
                        args = np.hstack((t.reshape(t.size, 1), np.tile(args[1:], t.size).reshape(t.size, len(args) - 1)))
                    return lamb(args)
                func = _wrapped_lamb
            except RuntimeError:
                import sympy
                func = sympy.lambdify(params, value)
            self.lambda_funcs[key] = func