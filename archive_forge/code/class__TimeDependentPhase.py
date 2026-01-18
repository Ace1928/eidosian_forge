import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
class _TimeDependentPhase(cirq.NoiseModel):

    def noisy_moment(self, moment, system_qubits):
        duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration(nanos=1))
        phase = duration.total_nanos() / 100.0
        yield (cirq.Y ** phase).on_each(system_qubits)
        yield moment