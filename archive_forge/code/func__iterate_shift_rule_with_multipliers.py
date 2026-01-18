import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
def _iterate_shift_rule_with_multipliers(rule, order, period):
    """Helper method to repeat a shift rule that includes multipliers multiple
    times along the same parameter axis for higher-order derivatives."""
    combined_rules = []
    for partial_rules in itertools.product(rule, repeat=order):
        c, m, s = np.stack(partial_rules).T
        cumul_shift = 0.0
        for _m, _s in zip(m, s):
            cumul_shift *= _m
            cumul_shift += _s
        if period is not None:
            cumul_shift = np.mod(cumul_shift + 0.5 * period, period) - 0.5 * period
        combined_rules.append(np.stack([np.prod(c), np.prod(m), cumul_shift]))
    return qml.math.stack(combined_rules)