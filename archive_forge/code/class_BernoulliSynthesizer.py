from typing import Optional, Sequence, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@frozen
class BernoulliSynthesizer(cirq_ft.PrepareOracle):
    """Synthesizes the state $sqrt(1 - p)|00..00> + sqrt(p)|11..11>$"""
    p: float
    nqubits: int

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('q', self.nqubits, 2),)

    def decompose_from_registers(self, context, q: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        theta = np.arccos(np.sqrt(1 - self.p))
        yield cirq.ry(2 * theta).on(q[0])
        yield [cirq.CNOT(q[0], q[i]) for i in range(1, len(q))]