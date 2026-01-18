from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
class DeterministicImbalancedStateSampler(cirq.Sampler):
    """A simple, deterministic mock sampler.
        Pretends to sample from a state vector with a 3:1 balance between the
        probabilities of the |0) and |1) state.
        """

    def run_sweep(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int=1) -> Sequence['cirq.Result']:
        results = np.zeros((repetitions, 1), dtype=bool)
        for idx in range(repetitions // 4):
            results[idx][0] = 1
        return [cirq.ResultDict(params=pr, measurements={'z': results}) for pr in cirq.study.to_resolvers(params)]