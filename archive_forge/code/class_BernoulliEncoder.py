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
class BernoulliEncoder(cirq_ft.SelectOracle):
    """Encodes Bernoulli random variable y0/y1 as $Enc|ii..i>|0> = |ii..i>|y_{i}>$ where i=0/1."""
    p: float
    y: Tuple[int, int]
    selection_bitsize: int
    target_bitsize: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return () if self.control_val is None else (cirq_ft.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('q', self.selection_bitsize, 2),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('t', self.target_bitsize),)

    def decompose_from_registers(self, context, q: Sequence[cirq.Qid], t: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        y0_bin = bit_tools.iter_bits(self.y[0], self.target_bitsize)
        y1_bin = bit_tools.iter_bits(self.y[1], self.target_bitsize)
        for y0, y1, tq in zip(y0_bin, y1_bin, t):
            if y0:
                yield cirq.X(tq).controlled_by(*q, control_values=[0] * self.selection_bitsize)
            if y1:
                yield cirq.X(tq).controlled_by(*q, control_values=[1] * self.selection_bitsize)

    def controlled(self, *args, **kwargs):
        cv = kwargs['control_values'][0]
        return BernoulliEncoder(self.p, self.y, self.selection_bitsize, self.target_bitsize, cv)

    @cached_property
    def mu(self) -> float:
        return self.p * self.y[1] + (1 - self.p) * self.y[0]

    @cached_property
    def s_square(self) -> float:
        return self.p * self.y[1] ** 2 + (1 - self.p) * self.y[0] ** 2