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
def _combine_shift_rules(rules):
    """Helper method to combine shift rules for multiple parameters into
    simultaneous multivariate shift rules."""
    combined_rules = []
    for partial_rules in itertools.product(*rules):
        c, *m, s = np.stack(partial_rules).T
        combined = np.concatenate([[np.prod(c)], *m, s])
        combined_rules.append(np.stack(combined))
    return np.stack(combined_rules)