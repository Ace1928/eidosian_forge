from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
@property
def _rng(self):
    if self._rng_generator is None:
        return np.random.default_rng()
    return self._rng_generator