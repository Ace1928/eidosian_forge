import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class ApplyXToIJKthQubit(cirq_ft.UnaryIterationGate):

    def __init__(self, target_shape: Tuple[int, int, int]):
        self._target_shape = target_shape

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return ()

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return tuple((cirq_ft.SelectionRegister('ijk'[i], (self._target_shape[i] - 1).bit_length(), self._target_shape[i]) for i in range(3)))

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return tuple(cirq_ft.Signature.build(t1=self._target_shape[0], t2=self._target_shape[1], t3=self._target_shape[2]))

    def nth_operation(self, context: cirq.DecompositionContext, control: cirq.Qid, i: int, j: int, k: int, t1: Sequence[cirq.Qid], t2: Sequence[cirq.Qid], t3: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield [cirq.CNOT(control, t1[i]), cirq.CNOT(control, t2[j]), cirq.CNOT(control, t3[k])]