import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
def _reorder_AmpPhaseFreq(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude and/or callable freq.

    Consolidates amplitude, phase and freq parameters if they are callable,
    and duplicates parameters since they will be passed to two operators in the Hamiltonian"""
    reordered_params = []
    coeff_idx = 0
    params_idx = 0
    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhaseAndFreq):
                is_callables = [coeff.phase_is_callable, coeff.amp_is_callable, coeff.freq_is_callable]
                num_callables = sum(is_callables)
                reordered_params.extend([params[params_idx:params_idx + num_callables]])
                coeff_idx += 1
                params_idx += num_callables
            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1
    return reordered_params