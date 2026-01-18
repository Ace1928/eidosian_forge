from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
class ExampleSimulationState(cirq.SimulationState):

    def __init__(self, fallback_result: Any=NotImplemented):
        super().__init__(prng=np.random.RandomState(), state=ExampleQuantumState())
        self.fallback_result = fallback_result

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool=True):
        return self.fallback_result