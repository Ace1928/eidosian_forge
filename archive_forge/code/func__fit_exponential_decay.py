import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _fit_exponential_decay(cycle_depths: np.ndarray, fidelities: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit an exponential model fidelity = a * layer_fid**x using nonlinear least squares.

    This uses `exponential_decay` as the function to fit with parameters `a` and `layer_fid`.

    Args:
        cycle_depths: The various depths at which fidelity was estimated. Each element is `x`
            in the fit expression.
        fidelities: The estimated fidelities for each cycle depth. Each element is `fidelity`
            in the fit expression.

    Returns:
        a: The first fit parameter that scales the exponential function, perhaps accounting for
            state prep and measurement (SPAM) error.
        layer_fid: The second fit parameters which serves as the base of the exponential.
        a_std: The standard deviation of the `a` parameter estimate.
        layer_fid_std: The standard deviation of the `layer_fid` parameter estimate.
    """
    cycle_depths = np.asarray(cycle_depths)
    fidelities = np.asarray(fidelities)
    positives = fidelities > 0
    if np.sum(positives) <= 1:
        return (0, 0, np.inf, np.inf)
    cycle_depths_pos = cycle_depths[positives]
    log_fidelities = np.log(fidelities[positives])
    slope, intercept, _, _, _ = stats.linregress(cycle_depths_pos, log_fidelities)
    layer_fid_0 = np.clip(np.exp(slope), 0, 1)
    a_0 = np.clip(np.exp(intercept), 0, 1)
    try:
        (a, layer_fid), pcov = optimize.curve_fit(exponential_decay, cycle_depths, fidelities, p0=(a_0, layer_fid_0), bounds=((0, 0), (1, 1)))
    except ValueError:
        return (0, 0, np.inf, np.inf)
    a_std, layer_fid_std = np.sqrt(np.diag(pcov))
    return (a, layer_fid, a_std, layer_fid_std)