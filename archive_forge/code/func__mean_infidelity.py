import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _mean_infidelity(angles):
    params = dict(zip(names, angles))
    if verbose:
        params_str = ''
        for name, val in params.items():
            params_str += f'{name:5s} = {val:7.3g} '
        print(f'Simulating with {params_str}')
    fids = benchmark_2q_xeb_fidelities(sampled_df, parameterized_circuits, cycle_depths, param_resolver=params, pool=pool)
    loss = 1 - fids['fidelity'].mean()
    if verbose:
        print(f'Loss: {loss:7.3g}', flush=True)
    return loss