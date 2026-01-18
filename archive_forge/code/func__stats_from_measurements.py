import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def _stats_from_measurements(bitstrings: np.ndarray, qubit_to_index: Dict['cirq.Qid', int], observable: 'cirq.PauliString', atol: float) -> Tuple[float, float]:
    """Return the mean and squared standard error of the mean for the given
    observable according to the measurements in `bitstrings`."""
    obs_vals = _obs_vals_from_measurements(bitstrings, qubit_to_index, observable, atol=atol)
    obs_mean = np.mean(obs_vals)
    obs_err = np.var(obs_vals, ddof=1) / len(obs_vals)
    return (obs_mean.item(), obs_err.item())