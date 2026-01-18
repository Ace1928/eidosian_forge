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
class ApplyXToLthQubit(cirq_ft.UnaryIterationGate):

    def __init__(self, selection_bitsize: int, target_bitsize: int, control_bitsize: int=1):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('control', self._control_bitsize),)

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self._selection_bitsize, self._target_bitsize),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('target', self._target_bitsize),)

    def nth_operation(self, context: cirq.DecompositionContext, control: cirq.Qid, selection: int, target: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(selection + 1)])