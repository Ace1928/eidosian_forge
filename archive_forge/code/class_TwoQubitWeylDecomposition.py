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
class TwoQubitWeylDecomposition:
    """Two-qubit Weyl decomposition.

    Decompose two-qubit unitary

    .. math::

        U = ({K_1}^l \\otimes {K_1}^r) e^{(i a XX + i b YY + i c ZZ)} ({K_2}^l \\otimes {K_2}^r)

    where

    .. math::

        U \\in U(4),~
        {K_1}^l, {K_1}^r, {K_2}^l, {K_2}^r \\in SU(2)

    and we stay in the "Weyl Chamber"

    .. math::

        \\pi /4 \\geq a \\geq b \\geq |c|

    This is an abstract factory class that instantiates itself as specialized subclasses based on
    the fidelity, such that the approximation error from specialization has an average gate fidelity
    at least as high as requested. The specialized subclasses have unique canonical representations
    thus avoiding problems of numerical stability.

    Passing non-None fidelity to specializations is treated as an assertion, raising QiskitError if
    forcing the specialization is more approximate than asserted.

    References:
        1. Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D. & Gambetta, J. M.,
           *Validating quantum computers using randomized model circuits*,
           `arXiv:1811.12926 [quant-ph] <https://arxiv.org/abs/1811.12926>`_
        2. B. Kraus, J. I. Cirac, *Optimal Creation of Entanglement Using a Two-Qubit Gate*,
           `arXiv:0011050 [quant-ph] <https://arxiv.org/abs/quant-ph/0011050>`_
        3. B. Drury, P. J. Love, *Constructive Quantum Shannon Decomposition from Cartan
           Involutions*, `arXiv:0806.4015 [quant-ph] <https://arxiv.org/abs/0806.4015>`_

    """
    a: float
    b: float
    c: float
    global_phase: float
    K1l: np.ndarray
    K2l: np.ndarray
    K1r: np.ndarray
    K2r: np.ndarray
    unitary_matrix: np.ndarray
    requested_fidelity: Optional[float]
    calculated_fidelity: float
    _original_decomposition: 'TwoQubitWeylDecomposition'
    _is_flipped_from_original: bool
    _default_1q_basis: ClassVar[str] = 'ZYZ'

    def __init_subclass__(cls, **kwargs):
        """Subclasses should be concrete, not factories.

        Make explicitly-instantiated subclass __new__  call base __new__ with fidelity=None"""
        super().__init_subclass__(**kwargs)
        cls.__new__ = lambda cls, *a, fidelity=None, **k: TwoQubitWeylDecomposition.__new__(cls, *a, fidelity=None, **k)

    @staticmethod
    def __new__(cls, unitary_matrix, *, fidelity=1.0 - 1e-09, _unpickling=False):
        """Perform the Weyl chamber decomposition, and optionally choose a specialized subclass."""
        if _unpickling:
            return super().__new__(cls)
        pi = np.pi
        pi2 = np.pi / 2
        pi4 = np.pi / 4
        U = np.array(unitary_matrix, dtype=complex, copy=True)
        detU = np.linalg.det(U)
        U *= detU ** (-0.25)
        global_phase = cmath.phase(detU) / 4
        Up = transform_to_magic_basis(U, reverse=True)
        M2 = Up.T.dot(Up)
        state = np.random.default_rng(2020)
        for _ in range(100):
            M2real = state.normal() * M2.real + state.normal() * M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=0, atol=1e-13):
                break
        else:
            raise QiskitError(f'TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: {U.tolist()}')
        d = -np.angle(D) / 2
        d[3] = -d[0] - d[1] - d[2]
        cs = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)
        cstemp = np.mod(cs, pi2)
        np.minimum(cstemp, pi2 - cstemp, cstemp)
        order = np.argsort(cstemp)[[1, 2, 0]]
        cs = cs[order]
        d[:3] = d[order]
        P[:, :3] = P[:, order]
        if np.real(np.linalg.det(P)) < 0:
            P[:, -1] = -P[:, -1]
        K1 = transform_to_magic_basis(Up @ P @ np.diag(np.exp(1j * d)))
        K2 = transform_to_magic_basis(P.T)
        K1l, K1r, phase_l = decompose_two_qubit_product_gate(K1)
        K2l, K2r, phase_r = decompose_two_qubit_product_gate(K2)
        global_phase += phase_l + phase_r
        K1l = K1l.copy()
        if cs[0] > pi2:
            cs[0] -= 3 * pi2
            K1l = K1l.dot(_ipy)
            K1r = K1r.dot(_ipy)
            global_phase += pi2
        if cs[1] > pi2:
            cs[1] -= 3 * pi2
            K1l = K1l.dot(_ipx)
            K1r = K1r.dot(_ipx)
            global_phase += pi2
        conjs = 0
        if cs[0] > pi4:
            cs[0] = pi2 - cs[0]
            K1l = K1l.dot(_ipy)
            K2r = _ipy.dot(K2r)
            conjs += 1
            global_phase -= pi2
        if cs[1] > pi4:
            cs[1] = pi2 - cs[1]
            K1l = K1l.dot(_ipx)
            K2r = _ipx.dot(K2r)
            conjs += 1
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if cs[2] > pi2:
            cs[2] -= 3 * pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if conjs == 1:
            cs[2] = pi2 - cs[2]
            K1l = K1l.dot(_ipz)
            K2r = _ipz.dot(K2r)
            global_phase += pi2
        if cs[2] > pi4:
            cs[2] -= pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase -= pi2
        a, b, c = (cs[1], cs[0], cs[2])
        od = super().__new__(TwoQubitWeylDecomposition)
        od.a = a
        od.b = b
        od.c = c
        od.K1l = K1l
        od.K1r = K1r
        od.K2l = K2l
        od.K2r = K2r
        od.global_phase = global_phase
        od.requested_fidelity = fidelity
        od.calculated_fidelity = 1.0
        od.unitary_matrix = np.array(unitary_matrix, dtype=complex, copy=True)
        od.unitary_matrix.setflags(write=False)
        od._original_decomposition = None
        od._is_flipped_from_original = False

        def is_close(ap, bp, cp):
            da, db, dc = (a - ap, b - bp, c - cp)
            tr = 4 * complex(math.cos(da) * math.cos(db) * math.cos(dc), math.sin(da) * math.sin(db) * math.sin(dc))
            fid = trace_to_fid(tr)
            return fid >= fidelity
        if fidelity is None:
            instance = super().__new__(TwoQubitWeylGeneral if cls is TwoQubitWeylDecomposition else cls)
        elif is_close(0, 0, 0):
            instance = super().__new__(TwoQubitWeylIdEquiv)
        elif is_close(pi4, pi4, pi4) or is_close(pi4, pi4, -pi4):
            instance = super().__new__(TwoQubitWeylSWAPEquiv)
        elif (lambda x: is_close(x, x, x))(_closest_partial_swap(a, b, c)):
            instance = super().__new__(TwoQubitWeylPartialSWAPEquiv)
        elif (lambda x: is_close(x, x, -x))(_closest_partial_swap(a, b, -c)):
            instance = super().__new__(TwoQubitWeylPartialSWAPFlipEquiv)
        elif is_close(a, 0, 0):
            instance = super().__new__(TwoQubitWeylControlledEquiv)
        elif is_close(pi4, pi4, c):
            instance = super().__new__(TwoQubitWeylMirrorControlledEquiv)
        elif is_close((a + b) / 2, (a + b) / 2, c):
            instance = super().__new__(TwoQubitWeylfSimaabEquiv)
        elif is_close(a, (b + c) / 2, (b + c) / 2):
            instance = super().__new__(TwoQubitWeylfSimabbEquiv)
        elif is_close(a, (b - c) / 2, (c - b) / 2):
            instance = super().__new__(TwoQubitWeylfSimabmbEquiv)
        else:
            instance = super().__new__(TwoQubitWeylGeneral)
        instance._original_decomposition = od
        return instance

    def __init__(self, unitary_matrix: list[list[complex]] | np.ndarray[complex], fidelity: float | None=None):
        """
        Args:
            unitary_matrix: The unitary to decompose.
            fidelity: The target fidelity of the decomposed operation.
        """
        del unitary_matrix
        od = self._original_decomposition
        self.a, self.b, self.c = (od.a, od.b, od.c)
        self.K1l, self.K1r = (od.K1l, od.K1r)
        self.K2l, self.K2r = (od.K2l, od.K2r)
        self.global_phase = od.global_phase
        self.unitary_matrix = od.unitary_matrix
        self.requested_fidelity = fidelity
        self._is_flipped_from_original = False
        self.specialize()
        if self._is_flipped_from_original:
            da, db, dc = (np.pi / 2 - od.a - self.a, od.b - self.b, -od.c - self.c)
            tr = 4 * complex(math.cos(da) * math.cos(db) * math.cos(dc), math.sin(da) * math.sin(db) * math.sin(dc))
        else:
            da, db, dc = (od.a - self.a, od.b - self.b, od.c - self.c)
            tr = 4 * complex(math.cos(da) * math.cos(db) * math.cos(dc), math.sin(da) * math.sin(db) * math.sin(dc))
        self.global_phase += cmath.phase(tr)
        self.calculated_fidelity = trace_to_fid(tr)
        if logger.isEnabledFor(logging.DEBUG):
            actual_fidelity = self.actual_fidelity()
            logger.debug('Requested fidelity: %s calculated fidelity: %s actual fidelity %s', self.requested_fidelity, self.calculated_fidelity, actual_fidelity)
            if abs(self.calculated_fidelity - actual_fidelity) > 1e-12:
                logger.warning('Requested fidelity different from actual by %s', self.calculated_fidelity - actual_fidelity)
        if self.requested_fidelity and self.calculated_fidelity + 1e-13 < self.requested_fidelity:
            raise QiskitError(f'{self.__class__.__name__}: calculated fidelity: {self.calculated_fidelity} is worse than requested fidelity: {self.requested_fidelity}.')

    def specialize(self):
        """Make changes to the decomposition to comply with any specialization."""
        raise NotImplementedError

    def circuit(self, *, euler_basis: str | None=None, simplify: bool=False, atol: float=DEFAULT_ATOL) -> QuantumCircuit:
        """Returns Weyl decomposition in circuit form."""
        if euler_basis is None:
            euler_basis = self._default_1q_basis
        oneq_decompose = OneQubitEulerDecomposer(euler_basis)
        c1l, c1r, c2l, c2r = (oneq_decompose(k, simplify=simplify, atol=atol) for k in (self.K1l, self.K1r, self.K2l, self.K2r))
        circ = QuantumCircuit(2, global_phase=self.global_phase)
        circ.compose(c2r, [0], inplace=True)
        circ.compose(c2l, [1], inplace=True)
        self._weyl_gate(simplify, circ, atol)
        circ.compose(c1r, [0], inplace=True)
        circ.compose(c1l, [1], inplace=True)
        return circ

    def _weyl_gate(self, simplify, circ: QuantumCircuit, atol):
        """Appends U_d(a, b, c) to the circuit.

        Can be overridden in subclasses for special cases"""
        if not simplify or abs(self.a) > atol:
            circ.rxx(-self.a * 2, 0, 1)
        if not simplify or abs(self.b) > atol:
            circ.ryy(-self.b * 2, 0, 1)
        if not simplify or abs(self.c) > atol:
            circ.rzz(-self.c * 2, 0, 1)

    def actual_fidelity(self, **kwargs) -> float:
        """Calculates the actual fidelity of the decomposed circuit to the input unitary."""
        circ = self.circuit(**kwargs)
        trace = np.trace(Operator(circ).data.T.conj() @ self.unitary_matrix)
        return trace_to_fid(trace)

    def __getnewargs_ex__(self):
        return ((self.unitary_matrix,), {'_unpickling': True})

    def __repr__(self):
        """Represent with enough precision to allow copy-paste debugging of all corner cases"""
        prefix = f'{type(self).__qualname__}.from_bytes('
        with io.BytesIO() as f:
            np.save(f, self.unitary_matrix, allow_pickle=False)
            b64 = base64.encodebytes(f.getvalue()).splitlines()
        b64ascii = [repr(x) for x in b64]
        b64ascii[-1] += ','
        pretty = [f'# {x.rstrip()}' for x in str(self).splitlines()]
        indent = '\n' + ' ' * 4
        lines = [prefix] + pretty + b64ascii + [f'requested_fidelity={self.requested_fidelity},', f'calculated_fidelity={self.calculated_fidelity},', f'actual_fidelity={self.actual_fidelity()},', f'abc={(self.a, self.b, self.c)})']
        return indent.join(lines)

    @classmethod
    def from_bytes(cls, bytes_in: bytes, *, requested_fidelity: float, **kwargs) -> 'TwoQubitWeylDecomposition':
        """Decode bytes into :class:`.TwoQubitWeylDecomposition`."""
        del kwargs
        b64 = base64.decodebytes(bytes_in)
        with io.BytesIO(b64) as f:
            arr = np.load(f, allow_pickle=False)
        return cls(arr, fidelity=requested_fidelity)

    def __str__(self):
        pre = f'{self.__class__.__name__}(\n\t'
        circ_indent = '\n\t'.join(self.circuit(simplify=True).draw('text').lines(-1))
        return f'{pre}{circ_indent}\n)'