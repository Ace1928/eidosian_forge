import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def _cpmg_sweep(num_pulses: List[int]):
    """Returns a sweep for a circuit created by _cpmg_circuit.

    The circuit in _cpmg_circuit parameterizes the pulses, so this function
    fills in the parameters for each pulse.  For instance, if we want 3 pulses,
    pulse_0, pulse_1, and pulse_2 should be 1 and the rest of the pulses should
    be 0.
    """
    pulse_points = []
    for n in range(max(num_pulses)):
        pulse_points.append(study.Points(f'pulse_{n}', [1 if p > n else 0 for p in num_pulses]))
    return study.Zip(*pulse_points)