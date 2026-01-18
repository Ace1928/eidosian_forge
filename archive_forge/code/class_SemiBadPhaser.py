import pytest
import numpy as np
import cirq
class SemiBadPhaser:

    def __init__(self, e):
        self.e = e

    def _unitary_(self):
        a1 = cirq.unitary(GoodPhaser(self.e[0]))
        a2 = cirq.unitary(BadPhaser(self.e[1]))
        return np.kron(a1, a2)

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        r = list(self.e)
        r[qubit_index] += phase_turns * 4
        return SemiBadPhaser(r)

    def _resolve_parameters_(self, resolver, recursive):
        return SemiBadPhaser([resolver.value_of(val, recursive) for val in self.e])