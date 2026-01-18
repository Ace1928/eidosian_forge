from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
class TwoQubitBasisDecomposer:
    """A class for decomposing 2-qubit unitaries into minimal number of uses of a 2-qubit
    basis gate.

    Args:
        gate: Two-qubit gate to be used in the KAK decomposition.
        basis_fidelity: Fidelity to be assumed for applications of KAK Gate. Defaults to ``1.0``.
        euler_basis: Basis string to be provided to :class:`.OneQubitEulerDecomposer` for 1Q synthesis.
            Valid options are [``'ZYZ'``, ``'ZXZ'``, ``'XYX'``, ``'U'``, ``'U3'``, ``'U1X'``,
            ``'PSX'``, ``'ZSX'``, ``'RR'``].
        pulse_optimize: If ``True``, try to do decomposition which minimizes
            local unitaries in between entangling gates. This will raise an exception if an
            optimal decomposition is not implemented. Currently, only [{CX, SX, RZ}] is known.
            If ``False``, don't attempt optimization. If ``None``, attempt optimization but don't raise
            if unknown.

    .. automethod:: __call__
    """

    def __init__(self, gate: Gate, basis_fidelity: float=1.0, euler_basis: str='U', pulse_optimize: bool | None=None):
        self.gate = gate
        self.basis_fidelity = basis_fidelity
        self.pulse_optimize = pulse_optimize
        basis = self.basis = TwoQubitWeylDecomposition(Operator(gate).data)
        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)
        self.is_supercontrolled = math.isclose(basis.a, np.pi / 4) and math.isclose(basis.c, 0.0)
        b = basis.b
        K11l = 1 / (1 + 1j) * np.array([[-1j * cmath.exp(-1j * b), cmath.exp(-1j * b)], [-1j * cmath.exp(1j * b), -cmath.exp(1j * b)]], dtype=complex)
        K11r = 1 / math.sqrt(2) * np.array([[1j * cmath.exp(-1j * b), -cmath.exp(-1j * b)], [cmath.exp(1j * b), -1j * cmath.exp(1j * b)]], dtype=complex)
        K12l = 1 / (1 + 1j) * np.array([[1j, 1j], [-1, 1]], dtype=complex)
        K12r = 1 / math.sqrt(2) * np.array([[1j, 1], [-1, -1j]], dtype=complex)
        K32lK21l = 1 / math.sqrt(2) * np.array([[1 + 1j * np.cos(2 * b), 1j * np.sin(2 * b)], [1j * np.sin(2 * b), 1 - 1j * np.cos(2 * b)]], dtype=complex)
        K21r = 1 / (1 - 1j) * np.array([[-1j * cmath.exp(-2j * b), cmath.exp(-2j * b)], [1j * cmath.exp(2j * b), cmath.exp(2j * b)]], dtype=complex)
        K22l = 1 / math.sqrt(2) * np.array([[1, -1], [1, 1]], dtype=complex)
        K22r = np.array([[0, 1], [-1, 0]], dtype=complex)
        K31l = 1 / math.sqrt(2) * np.array([[cmath.exp(-1j * b), cmath.exp(-1j * b)], [-cmath.exp(1j * b), cmath.exp(1j * b)]], dtype=complex)
        K31r = 1j * np.array([[cmath.exp(1j * b), 0], [0, -cmath.exp(-1j * b)]], dtype=complex)
        K32r = 1 / (1 - 1j) * np.array([[cmath.exp(1j * b), -cmath.exp(-1j * b)], [-1j * cmath.exp(1j * b), -1j * cmath.exp(-1j * b)]], dtype=complex)
        k1ld = basis.K1l.T.conj()
        k1rd = basis.K1r.T.conj()
        k2ld = basis.K2l.T.conj()
        k2rd = basis.K2r.T.conj()
        self.u0l = K31l.dot(k1ld)
        self.u0r = K31r.dot(k1rd)
        self.u1l = k2ld.dot(K32lK21l).dot(k1ld)
        self.u1ra = k2rd.dot(K32r)
        self.u1rb = K21r.dot(k1rd)
        self.u2la = k2ld.dot(K22l)
        self.u2lb = K11l.dot(k1ld)
        self.u2ra = k2rd.dot(K22r)
        self.u2rb = K11r.dot(k1rd)
        self.u3l = k2ld.dot(K12l)
        self.u3r = k2rd.dot(K12r)
        self.q0l = K12l.T.conj().dot(k1ld)
        self.q0r = K12r.T.conj().dot(_ipz).dot(k1rd)
        self.q1la = k2ld.dot(K11l.T.conj())
        self.q1lb = K11l.dot(k1ld)
        self.q1ra = k2rd.dot(_ipz).dot(K11r.T.conj())
        self.q1rb = K11r.dot(k1rd)
        self.q2l = k2ld.dot(K12l)
        self.q2r = k2rd.dot(K12r)
        if not self.is_supercontrolled:
            warnings.warn('Only know how to decompose properly for supercontrolled basis gate. This gate is ~Ud({}, {}, {})'.format(basis.a, basis.b, basis.c), stacklevel=2)
        self.decomposition_fns = [self.decomp0, self.decomp1, self.decomp2_supercontrolled, self.decomp3_supercontrolled]
        self._rqc = None

    def traces(self, target):
        """
        Give the expected traces :math:`\\Big\\vert\\text{Tr}(U \\cdot U_\\text{target}^{\\dag})\\Big\\vert`
        for a different number of basis gates.
        """
        ta, tb, tc = (target.a, target.b, target.c)
        bb = self.basis.b
        return [4 * complex(math.cos(ta) * math.cos(tb) * math.cos(tc), math.sin(ta) * math.sin(tb) * math.sin(tc)), 4 * complex(math.cos(math.pi / 4 - ta) * math.cos(bb - tb) * math.cos(tc), math.sin(math.pi / 4 - ta) * math.sin(bb - tb) * math.sin(tc)), 4 * math.cos(tc), 4]

    @staticmethod
    def decomp0(target):
        """
        Decompose target :math:`\\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
        Result :math:`U_r` has trace:

        .. math::

            \\Big\\vert\\text{Tr}(U_r\\cdot U_\\text{target}^{\\dag})\\Big\\vert =
            4\\Big\\vert (\\cos(x)\\cos(y)\\cos(z)+ j \\sin(x)\\sin(y)\\sin(z)\\Big\\vert

        which is optimal for all targets and bases
        """
        U0l = target.K1l.dot(target.K2l)
        U0r = target.K1r.dot(target.K2r)
        return (U0r, U0l)

    def decomp1(self, target):
        """Decompose target :math:`\\sim U_d(x, y, z)` with :math:`1` use of the basis gate
        :math:`\\sim U_d(a, b, c)`.
        Result :math:`U_r` has trace:

        .. math::

            \\Big\\vert\\text{Tr}(U_r \\cdot U_\\text{target}^{\\dag})\\Big\\vert =
            4\\Big\\vert \\cos(x-a)\\cos(y-b)\\cos(z-c) + j \\sin(x-a)\\sin(y-b)\\sin(z-c)\\Big\\vert

        which is optimal for all targets and bases with ``z==0`` or ``c==0``.
        """
        U0l = target.K1l.dot(self.basis.K1l.T.conj())
        U0r = target.K1r.dot(self.basis.K1r.T.conj())
        U1l = self.basis.K2l.T.conj().dot(target.K2l)
        U1r = self.basis.K2r.T.conj().dot(target.K2r)
        return (U1r, U1l, U0r, U0l)

    def decomp2_supercontrolled(self, target):
        """
        Decompose target :math:`\\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.

        For supercontrolled basis :math:`\\sim U_d(\\pi/4, b, 0)`, all b, result :math:`U_r` has trace

        .. math::

            \\Big\\vert\\text{Tr}(U_r \\cdot U_\\text{target}^\\dag) \\Big\\vert = 4\\cos(z)

        which is the optimal approximation for basis of CNOT-class :math:`\\sim U_d(\\pi/4, 0, 0)`
        or DCNOT-class :math:`\\sim U_d(\\pi/4, \\pi/4, 0)` and any target. It may
        be sub-optimal for :math:`b \\neq 0` (i.e. there exists an exact decomposition for any target
        using :math:`B \\sim U_d(\\pi/4, \\pi/8, 0)`, but it may not be this decomposition).
        This is an exact decomposition for supercontrolled basis and target :math:`\\sim U_d(x, y, 0)`.
        No guarantees for non-supercontrolled basis.
        """
        U0l = target.K1l.dot(self.q0l)
        U0r = target.K1r.dot(self.q0r)
        U1l = self.q1la.dot(rz_array(-2 * target.a)).dot(self.q1lb)
        U1r = self.q1ra.dot(rz_array(2 * target.b)).dot(self.q1rb)
        U2l = self.q2l.dot(target.K2l)
        U2r = self.q2r.dot(target.K2r)
        return (U2r, U2l, U1r, U1l, U0r, U0l)

    def decomp3_supercontrolled(self, target):
        """
        Decompose target with :math:`3` uses of the basis.
        This is an exact decomposition for supercontrolled basis :math:`\\sim U_d(\\pi/4, b, 0)`, all b,
        and any target. No guarantees for non-supercontrolled basis.
        """
        U0l = target.K1l.dot(self.u0l)
        U0r = target.K1r.dot(self.u0r)
        U1l = self.u1l
        U1r = self.u1ra.dot(rz_array(-2 * target.c)).dot(self.u1rb)
        U2l = self.u2la.dot(rz_array(-2 * target.a)).dot(self.u2lb)
        U2r = self.u2ra.dot(rz_array(2 * target.b)).dot(self.u2rb)
        U3l = self.u3l.dot(target.K2l)
        U3r = self.u3r.dot(target.K2r)
        return (U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l)

    def __call__(self, unitary: Operator | np.ndarray, basis_fidelity: float | None=None, approximate: bool=True, *, _num_basis_uses: int | None=None) -> QuantumCircuit:
        """Decompose a two-qubit ``unitary`` over fixed basis and :math:`SU(2)` using the best
        approximation given that each basis application has a finite ``basis_fidelity``.

        Args:
            unitary (Operator or ndarray): :math:`4 \\times 4` unitary to synthesize.
            basis_fidelity (float or None): Fidelity to be assumed for applications of KAK Gate.
                If given, overrides ``basis_fidelity`` given at init.
            approximate (bool): Approximates if basis fidelities are less than 1.0.
            _num_basis_uses (int): force a particular approximation by passing a number in [0, 3].

        Returns:
            QuantumCircuit: Synthesized quantum circuit.

        Raises:
            QiskitError: if ``pulse_optimize`` is True but we don't know how to do it.
        """
        basis_fidelity = basis_fidelity or self.basis_fidelity
        if approximate is False:
            basis_fidelity = 1.0
        unitary = np.asarray(unitary, dtype=complex)
        target_decomposed = TwoQubitWeylDecomposition(unitary)
        traces = self.traces(target_decomposed)
        expected_fidelities = [trace_to_fid(traces[i]) * basis_fidelity ** i for i in range(4)]
        best_nbasis = int(np.argmax(expected_fidelities))
        if _num_basis_uses is not None:
            best_nbasis = _num_basis_uses
        decomposition = self.decomposition_fns[best_nbasis](target_decomposed)
        try:
            if self.pulse_optimize in {None, True}:
                return_circuit = self._pulse_optimal_chooser(best_nbasis, decomposition, target_decomposed)
                if return_circuit:
                    return return_circuit
        except QiskitError:
            if self.pulse_optimize:
                raise
        q = QuantumRegister(2)
        decomposition_euler = [self._decomposer1q._decompose(x) for x in decomposition]
        return_circuit = QuantumCircuit(q)
        return_circuit.global_phase = target_decomposed.global_phase
        return_circuit.global_phase -= best_nbasis * self.basis.global_phase
        if best_nbasis == 2:
            return_circuit.global_phase += np.pi
        for i in range(best_nbasis):
            return_circuit.compose(decomposition_euler[2 * i], [q[0]], inplace=True)
            return_circuit.compose(decomposition_euler[2 * i + 1], [q[1]], inplace=True)
            return_circuit.append(self.gate, [q[0], q[1]])
        return_circuit.compose(decomposition_euler[2 * best_nbasis], [q[0]], inplace=True)
        return_circuit.compose(decomposition_euler[2 * best_nbasis + 1], [q[1]], inplace=True)
        return return_circuit

    def _pulse_optimal_chooser(self, best_nbasis, decomposition, target_decomposed) -> QuantumCircuit:
        """Determine method to find pulse optimal circuit. This method may be
        removed once a more general approach is used.

        Returns:
            QuantumCircuit: pulse optimal quantum circuit.
            None: Probably ``nbasis==1`` and original circuit is fine.

        Raises:
            QiskitError: Decomposition for selected basis not implemented.
        """
        circuit = None
        if self.pulse_optimize and best_nbasis in {0, 1}:
            return None
        elif self.pulse_optimize and best_nbasis > 3:
            raise QiskitError(f'Unexpected number of entangling gates ({best_nbasis}) in decomposition.')
        if self._decomposer1q.basis in {'ZSX', 'ZSXX'}:
            if isinstance(self.gate, CXGate):
                if best_nbasis == 3:
                    circuit = self._get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
                elif best_nbasis == 2:
                    circuit = self._get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed)
            else:
                raise QiskitError('pulse_optimizer currently only works with CNOT entangling gate')
        else:
            raise QiskitError(f'"pulse_optimize" currently only works with ZSX basis ({self._decomposer1q.basis} used)')
        return circuit

    def _get_sx_vz_2cx_efficient_euler(self, decomposition, target_decomposed):
        """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        two CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
        end of decomposition. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
        best_nbasis = 2
        num_1q_uni = len(decomposition)
        euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
        euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
        global_phase = 0.0
        zxz_decomposer = OneQubitEulerDecomposer('ZXZ')
        for iqubit, decomp in enumerate(decomposition[0::2]):
            euler_angles = zxz_decomposer.angles_and_phase(decomp)
            euler_q0[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        xzx_decomposer = OneQubitEulerDecomposer('XZX')
        for iqubit, decomp in enumerate(decomposition[1::2]):
            euler_angles = xzx_decomposer.angles_and_phase(decomp)
            euler_q1[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        qc = QuantumCircuit(2)
        qc.global_phase = target_decomposed.global_phase
        qc.global_phase -= best_nbasis * self.basis.global_phase
        qc.global_phase += global_phase
        circ = QuantumCircuit(1)
        circ.rz(euler_q0[0][0], 0)
        circ.rx(euler_q0[0][1], 0)
        circ.rz(euler_q0[0][2] + euler_q0[1][0] + math.pi / 2, 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)
        circ = QuantumCircuit(1)
        circ.rx(euler_q1[0][0], 0)
        circ.rz(euler_q1[0][1], 0)
        circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)
        qc.cx(0, 1)
        qc.sx(0)
        qc.rz(euler_q0[1][1] - math.pi, 0)
        qc.sx(0)
        qc.rz(euler_q1[1][1], 1)
        qc.global_phase += math.pi / 2
        qc.cx(0, 1)
        circ = QuantumCircuit(1)
        circ.rz(euler_q0[1][2] + euler_q0[2][0] + math.pi / 2, 0)
        circ.rx(euler_q0[2][1], 0)
        circ.rz(euler_q0[2][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)
        circ = QuantumCircuit(1)
        circ.rx(euler_q1[1][2] + euler_q1[2][0], 0)
        circ.rz(euler_q1[2][1], 0)
        circ.rx(euler_q1[2][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)
        return qc

    def _get_sx_vz_3cx_efficient_euler(self, decomposition, target_decomposed):
        """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        three CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
        end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
        a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
        best_nbasis = 3
        num_1q_uni = len(decomposition)
        euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
        euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
        global_phase = 0.0
        atol = 1e-10
        zxz_decomposer = OneQubitEulerDecomposer('ZXZ')
        for iqubit, decomp in enumerate(decomposition[0::2]):
            euler_angles = zxz_decomposer.angles_and_phase(decomp)
            euler_q0[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        xzx_decomposer = OneQubitEulerDecomposer('XZX')
        for iqubit, decomp in enumerate(decomposition[1::2]):
            euler_angles = xzx_decomposer.angles_and_phase(decomp)
            euler_q1[iqubit, [1, 2, 0]] = euler_angles[:3]
            global_phase += euler_angles[3]
        qc = QuantumCircuit(2)
        qc.global_phase = target_decomposed.global_phase
        qc.global_phase -= best_nbasis * self.basis.global_phase
        qc.global_phase += global_phase
        x12 = euler_q0[1][2] + euler_q0[2][0]
        x12_isNonZero = not math.isclose(x12, 0, abs_tol=atol)
        x12_isOddMult = None
        x12_isPiMult = math.isclose(math.sin(x12), 0, abs_tol=atol)
        if x12_isPiMult:
            x12_isOddMult = math.isclose(math.cos(x12), -1, abs_tol=atol)
            x12_phase = math.pi * math.cos(x12)
        x02_add = x12 - euler_q0[1][0]
        x12_isHalfPi = math.isclose(x12, math.pi / 2, abs_tol=atol)
        circ = QuantumCircuit(1)
        circ.rz(euler_q0[0][0], 0)
        circ.rx(euler_q0[0][1], 0)
        if x12_isNonZero and x12_isPiMult:
            circ.rz(euler_q0[0][2] - x02_add, 0)
        else:
            circ.rz(euler_q0[0][2] + euler_q0[1][0], 0)
        circ.h(0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)
        circ = QuantumCircuit(1)
        circ.rx(euler_q1[0][0], 0)
        circ.rz(euler_q1[0][1], 0)
        circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
        circ.h(0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)
        qc.cx(1, 0)
        if x12_isPiMult:
            if x12_isNonZero:
                qc.global_phase += x12_phase
            if x12_isNonZero and x12_isOddMult:
                qc.rz(-euler_q0[1][1], 0)
            else:
                qc.rz(euler_q0[1][1], 0)
                qc.global_phase += math.pi
        if x12_isHalfPi:
            qc.sx(0)
            qc.global_phase -= math.pi / 4
        elif x12_isNonZero and (not x12_isPiMult):
            if self.pulse_optimize is None:
                qc.compose(self._decomposer1q(Operator(RXGate(x12)).data), [0], inplace=True)
            else:
                raise QiskitError('possible non-pulse-optimal decomposition encountered')
        if math.isclose(euler_q1[1][1], math.pi / 2, abs_tol=atol):
            qc.sx(1)
            qc.global_phase -= math.pi / 4
        elif self.pulse_optimize is None:
            qc.compose(self._decomposer1q(Operator(RXGate(euler_q1[1][1])).data), [1], inplace=True)
        else:
            raise QiskitError('possible non-pulse-optimal decomposition encountered')
        qc.rz(euler_q1[1][2] + euler_q1[2][0], 1)
        qc.cx(1, 0)
        qc.rz(euler_q0[2][1], 0)
        if math.isclose(euler_q1[2][1], math.pi / 2, abs_tol=atol):
            qc.sx(1)
            qc.global_phase -= math.pi / 4
        elif self.pulse_optimize is None:
            qc.compose(self._decomposer1q(Operator(RXGate(euler_q1[2][1])).data), [1], inplace=True)
        else:
            raise QiskitError('possible non-pulse-optimal decomposition encountered')
        qc.cx(1, 0)
        circ = QuantumCircuit(1)
        circ.h(0)
        circ.rz(euler_q0[2][2] + euler_q0[3][0], 0)
        circ.rx(euler_q0[3][1], 0)
        circ.rz(euler_q0[3][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [0], inplace=True)
        circ = QuantumCircuit(1)
        circ.h(0)
        circ.rx(euler_q1[2][2] + euler_q1[3][0], 0)
        circ.rz(euler_q1[3][1], 0)
        circ.rx(euler_q1[3][2], 0)
        qceuler = self._decomposer1q(Operator(circ).data)
        qc.compose(qceuler, [1], inplace=True)
        if cmath.isclose(target_decomposed.unitary_matrix[0, 0], -Operator(qc).data[0, 0], abs_tol=atol):
            qc.global_phase += math.pi
        return qc

    def num_basis_gates(self, unitary):
        """Computes the number of basis gates needed in
        a decomposition of input unitary
        """
        return two_qubit_decompose._num_basis_gates(self.basis.b, self.basis_fidelity, np.asarray(unitary, dtype=complex))