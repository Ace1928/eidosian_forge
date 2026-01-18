import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _ref_simulate_2q_xeb_circuits(circuits: Sequence['cirq.Circuit'], cycle_depths: Sequence[int], param_resolver: 'cirq.ParamResolverOrSimilarType'=None, pool: Optional['multiprocessing.pool.Pool']=None):
    """Reference implementation for `simulate_2q_xeb_circuits` that
    does each circuit independently instead of using intermediate states.

    You can also try editing the helper function to use QSimSimulator() for
    benchmarking. This simulator does not support intermediate states, so
    you can't use it with the new functionality.
    https://github.com/quantumlib/qsim/issues/101
    """
    tasks = []
    for cycle_depth in cycle_depths:
        for circuit_i, circuit in enumerate(circuits):
            tasks += [{'circuit_i': circuit_i, 'cycle_depth': cycle_depth, 'circuit': circuit, 'param_resolver': param_resolver}]
    if pool is not None:
        records = pool.map(_ref_simulate_2q_xeb_circuit, tasks, chunksize=4)
    else:
        records = [_ref_simulate_2q_xeb_circuit(record) for record in tasks]
    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth']).sort_index()