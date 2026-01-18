from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, apply_gate_to_lth_target, arithmetic_gates
from cirq_ft.algos import prepare_uniform_superposition as prep_u
from cirq_ft.algos import (
@attr.frozen
class PrepareHubbard(select_and_prepare.PrepareOracle):
    """The PREPARE operation optimized for the 2D Hubbard model.

    In contrast to the arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` parameter.

    The circuit for implementing $\\textit{PREPARE}_{Hubbard}$ has a T-cost of $O(log(N)$
    and uses $O(1)$ single qubit rotations.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        t: coefficient for hopping terms in the Hubbard model hamiltonian.
        mu: coefficient for single body Z term and two-body ZZ terms in the Hubbard model
            hamiltonian.

    Signature:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        junk: Temporary Work space.

    References:
        Section V. and Fig. 20 of https://arxiv.org/abs/1805.03662.
    """
    x_dim: int
    y_dim: int
    t: int
    mu: int

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError('Currently only supports the case where x_dim=y_dim.')

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return (infra.SelectionRegister('U', 1, 2), infra.SelectionRegister('V', 1, 2), infra.SelectionRegister('p_x', (self.x_dim - 1).bit_length(), self.x_dim), infra.SelectionRegister('p_y', (self.y_dim - 1).bit_length(), self.y_dim), infra.SelectionRegister('alpha', 1, 2), infra.SelectionRegister('q_x', (self.x_dim - 1).bit_length(), self.x_dim), infra.SelectionRegister('q_y', (self.y_dim - 1).bit_length(), self.y_dim), infra.SelectionRegister('beta', 1, 2))

    @cached_property
    def junk_registers(self) -> Tuple[infra.Register, ...]:
        return (infra.Register('temp', 2),)

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        p_x, p_y, q_x, q_y = (quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y'])
        U, V, alpha, beta = (quregs['U'], quregs['V'], quregs['alpha'], quregs['beta'])
        temp = quregs['temp']
        N = self.x_dim * self.y_dim * 2
        qlambda = 2 * N * self.t + N * self.mu // 2
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(self.t * N / qlambda))).on(*V)
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)
        yield prep_u.PrepareUniformSuperposition(self.x_dim).on_registers(controls=[], target=p_x)
        yield prep_u.PrepareUniformSuperposition(self.y_dim).on_registers(controls=[], target=p_y)
        yield cirq.H.on_each(*temp)
        yield cirq.CNOT(*U, *V)
        yield cirq.X(*beta)
        yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]
        yield cirq.Circuit(cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])]))
        yield swap_network.MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)
        yield arithmetic_gates.AddMod(len(q_x), self.x_dim, add_val=1, cv=[0, 0]).on(*U, *V, *q_x)
        yield swap_network.MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)
        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield and_gate.And(cv=(0, 0, 1)).on_registers(ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target)
        yield swap_network.MultiTargetCSwap.make_on(control=and_target, target_x=[*p_x, *p_y, *alpha], target_y=[*q_x, *q_y, *beta])
        yield and_gate.And(cv=(0, 0, 1), adjoint=True).on_registers(ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target)
        context.qubit_manager.qfree([*and_anc, *and_target])

    def __repr__(self) -> str:
        return f'cirq_ft.PrepareHubbard({self.x_dim}, {self.y_dim}, {self.t}, {self.mu})'