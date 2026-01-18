from typing import Any, List, Sequence, Tuple
import cirq
import pytest
from pyquil import Program
from pyquil.api import QuantumComputer
import numpy as np
from pyquil.gates import MEASURE, RX, X, DECLARE, H, CNOT
from cirq_rigetti import RigettiQCSService
from typing_extensions import Protocol
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
class _ResultBuilder(Protocol):

    def __call__(self, mock_qpu_implementer: Any, circuit: cirq.Circuit, sweepable: cirq.Sweepable, *, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], List[cirq.ParamResolver]]:
        pass