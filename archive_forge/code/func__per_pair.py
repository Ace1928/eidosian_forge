import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _per_pair(f1):
    a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(f1['cycle_depth'], f1['fidelity'])
    record = {'a': a, 'layer_fid': layer_fid, 'cycle_depths': f1['cycle_depth'].values, 'fidelities': f1['fidelity'].values, 'a_std': a_std, 'layer_fid_std': layer_fid_std}
    return pd.Series(record)