from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline
def _is_deprecated_data_format(data) -> bool:
    if not isinstance(data, list):
        data = [data]
    for dat in data:
        if isinstance(dat, (QuasiDistribution, ProbDistribution)) or isinstance(next(iter(dat.values())), float):
            return True
    return False