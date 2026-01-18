import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
class GradualDecay(cirq.NoiseModel):

    def __init__(self, t1: float):
        self.t1 = t1

    def noisy_moment(self, moment, system_qubits):
        duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration(nanos=0))
        if duration > cirq.Duration(nanos=0):
            return cirq.amplitude_damp(1 - np.exp(-duration.total_nanos() / self.t1)).on_each(system_qubits)
        return moment