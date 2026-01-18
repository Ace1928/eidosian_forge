from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline
def _keep_largest_items(execution, number_to_keep):
    """Keep only the largest values in a dictionary, and sum the rest into a new key 'rest'."""
    sorted_counts = sorted(execution.items(), key=lambda p: p[1])
    rest = sum((count for key, count in sorted_counts[:-number_to_keep]))
    return dict(sorted_counts[-number_to_keep:], rest=rest)