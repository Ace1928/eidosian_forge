from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline
def _unify_labels(data):
    """Make all dictionaries in data have the same set of keys, using 0 for missing values."""
    data = tuple(data)
    all_labels = set().union(*(execution.keys() for execution in data))
    base = {label: 0 for label in all_labels}
    out = []
    for execution in data:
        new_execution = base.copy()
        new_execution.update(execution)
        out.append(new_execution)
    return out