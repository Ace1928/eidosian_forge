import pytest
import numpy as np
import cirq
class BadPhaser:

    def __init__(self, e):
        self.e = e

    def _unitary_(self):
        return np.array([[0, 1j ** (-(self.e * 2))], [1j ** self.e, 0]])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return BadPhaser(self.e + phase_turns * 4)

    def _resolve_parameters_(self, resolver, recursive):
        return BadPhaser(resolver.value_of(self.e, recursive))