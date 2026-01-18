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
@functools.lru_cache(maxsize=None)
def _is_amplitude_valid(envelope_lam: Callable, time: tuple[float, ...], *fargs: float) -> bool | np.bool_:
    """A helper function to validate maximum amplitude limit.

    Result is cached for better performance.

    Args:
        envelope_lam: The SymbolicPulse's lambdified envelope_lam expression.
        time: The SymbolicPulse's time array, given as a tuple for hashability.
        fargs: The arguments for the lambdified envelope_lam, as given by `_get_expression_args`,
            except for the time array.

    Returns:
        Return True if no sample point exceeds 1.0 in absolute value.
    """
    time = np.asarray(time, dtype=float)
    samples_norm = np.abs(envelope_lam(time, *fargs))
    epsilon = 1e-07
    return np.all(samples_norm < 1.0 + epsilon)