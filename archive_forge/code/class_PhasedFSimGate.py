import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
@value.value_equality(approximate=True)
class PhasedFSimGate(gate_features.InterchangeableQubitsGate, raw_types.Gate):
    """General excitation-preserving two-qubit gate.

    The unitary matrix of PhasedFSimGate(θ, ζ, χ, γ, φ) is:

    $$
    \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & e^{-i \\gamma - i \\zeta} \\cos(\\theta) & -i e^{-i \\gamma + i\\chi} \\sin(\\theta) & 0 \\\\
        0 & -i e^{-i \\gamma - i \\chi} \\sin(\\theta) & e^{-i \\gamma + i \\zeta} \\cos(\\theta) & 0 \\\\
        0 & 0 & 0 & e^{-2i \\gamma - i \\phi}
    \\end{bmatrix}
    $$

    This parametrization follows eq (18) in https://arxiv.org/abs/2010.07965.
    See also eq (43) in https://arxiv.org/abs/1910.11333 for an older variant
    which uses the same θ and φ parameters, but has three phase angles that
    have different names and opposite sign. Specifically, ∆+ angle corresponds
    to -γ, ∆- corresponds to -ζ and ∆-,off corresponds to -χ.

    Another useful parametrization of PhasedFSimGate is based on the fact that
    the gate is equivalent up to global phase to the following circuit:

        0: ───Rz(α0)───FSim(θ, φ)───Rz(β0)───
                       │
        1: ───Rz(α1)───FSim(θ, φ)───Rz(β1)───

    where α0 and α1 are Rz angles to be applied before the core FSimGate,
    β0 and β1 are Rz angles to be applied after FSimGate and θ and φ specify
    the core FSimGate. Use the static factory function from_fsim_rz to
    instantiate the gate using this parametrization.

    Note that the θ and φ parameters in the two parametrizations are the same.

    The matrix above is block diagonal where the middle block may be any
    element of U(2) and the bottom right block may be any element of U(1).
    Consequently, five real parameters are required to specify an instance
    of PhasedFSimGate. Therefore, the second parametrization is not injective.
    Indeed, for any angle δ

        cirq.PhasedFSimGate.from_fsim_rz(θ, φ, (α0, α1), (β0, β1))

    and

        cirq.PhasedFSimGate.from_fsim_rz(θ, φ,
                                         (α0 + δ, α1 + δ),
                                         (β0 - δ, β1 - δ))

    specify the same gate and therefore the two instances will compare as
    equal up to numerical error. Another consequence of the non-injective
    character of the second parametrization is the fact that the properties
    rz_angles_before and rz_angles_after may return different Rz angles
    than the ones used in the call to from_fsim_rz.

    This gate is generally not symmetric under exchange of qubits. It becomes
    symmetric if both of the following conditions are satisfied:
     * ζ = kπ or θ = π/2 + lπ for k and l integers,
     * χ = kπ or θ = lπ for k and l integers.
    """

    def __init__(self, theta: 'cirq.TParamVal', zeta: 'cirq.TParamVal'=0.0, chi: 'cirq.TParamVal'=0.0, gamma: 'cirq.TParamVal'=0.0, phi: 'cirq.TParamVal'=0.0) -> None:
        """Inits PhasedFSimGate.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            zeta: One of the phase angles, in radians. See class
                docstring above for details.
            chi: One of the phase angles, in radians.
                See class docstring above for details.
            gamma: One of the phase angles, in radians. See class
                docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
        """
        self._theta = _canonicalize(theta)
        self._zeta = _canonicalize(zeta)
        self._chi = _canonicalize(chi)
        self._gamma = _canonicalize(gamma)
        self._phi = _canonicalize(phi)

    @property
    def theta(self) -> 'cirq.TParamVal':
        return self._theta

    @property
    def zeta(self) -> 'cirq.TParamVal':
        return self._zeta

    @property
    def chi(self) -> 'cirq.TParamVal':
        return self._chi

    @property
    def gamma(self) -> 'cirq.TParamVal':
        return self._gamma

    @property
    def phi(self) -> 'cirq.TParamVal':
        return self._phi

    @staticmethod
    def from_fsim_rz(theta: 'cirq.TParamVal', phi: 'cirq.TParamVal', rz_angles_before: Tuple['cirq.TParamVal', 'cirq.TParamVal'], rz_angles_after: Tuple['cirq.TParamVal', 'cirq.TParamVal']) -> 'PhasedFSimGate':
        """Creates PhasedFSimGate using an alternate parametrization.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
            rz_angles_before: 2-tuple of phase angles to apply to each qubit
                before the core FSimGate. See class docstring for details.
            rz_angles_after: 2-tuple of phase angles to apply to each qubit
                after the core FSimGate. See class docstring for details.
        """
        b0, b1 = rz_angles_before
        a0, a1 = rz_angles_after
        gamma = (-b0 - b1 - a0 - a1) / 2.0
        zeta = (b0 - b1 + a0 - a1) / 2.0
        chi = (b0 - b1 - a0 + a1) / 2.0
        return PhasedFSimGate(theta, zeta, chi, gamma, phi)

    @property
    def rz_angles_before(self) -> Tuple['cirq.TParamVal', 'cirq.TParamVal']:
        """Returns 2-tuple of phase angles applied to qubits before FSimGate."""
        b0 = (-self.gamma + self.zeta + self.chi) / 2.0
        b1 = (-self.gamma - self.zeta - self.chi) / 2.0
        return (b0, b1)

    @property
    def rz_angles_after(self) -> Tuple['cirq.TParamVal', 'cirq.TParamVal']:
        """Returns 2-tuple of phase angles applied to qubits after FSimGate."""
        a0 = (-self.gamma + self.zeta - self.chi) / 2.0
        a1 = (-self.gamma - self.zeta + self.chi) / 2.0
        return (a0, a1)

    def _zeta_insensitive(self) -> bool:
        return _half_pi_mod_pi(self.theta)

    def _chi_insensitive(self) -> bool:
        return _zero_mod_pi(self.theta)

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        if (_zero_mod_pi(self.zeta) or self._zeta_insensitive()) and (_zero_mod_pi(self.chi) or self._chi_insensitive()):
            return 0
        return index

    def _value_equality_values_(self) -> Any:
        if self._zeta_insensitive():
            return (self.theta, 0.0, self.chi, self.gamma, self.phi)
        if self._chi_insensitive():
            return (self.theta, self.zeta, 0.0, self.gamma, self.phi)
        return (self.theta, self.zeta, self.chi, self.gamma, self.phi)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.theta) or cirq.is_parameterized(self.zeta) or cirq.is_parameterized(self.chi) or cirq.is_parameterized(self.gamma) or cirq.is_parameterized(self.phi)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        f1 = cmath.exp(-1j * self.gamma - 1j * self.zeta)
        f2 = cmath.exp(-1j * self.gamma + 1j * self.chi)
        f3 = cmath.exp(-1j * self.gamma - 1j * self.chi)
        f4 = cmath.exp(-1j * self.gamma + 1j * self.zeta)
        f5 = cmath.exp(-2j * self.gamma)
        return np.array([[1, 0, 0, 0], [0, f1 * a, f2 * b, 0], [0, f3 * b, f4 * a, 0], [0, 0, 0, f5 * c]])

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'cirq.PhasedFSimGate':
        return PhasedFSimGate(protocols.resolve_parameters(self.theta, resolver, recursive), protocols.resolve_parameters(self.zeta, resolver, recursive), protocols.resolve_parameters(self.chi, resolver, recursive), protocols.resolve_parameters(self.gamma, resolver, recursive), protocols.resolve_parameters(self.phi, resolver, recursive))

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        oi = args.subspace_index(1)
        io = args.subspace_index(2)
        ii = args.subspace_index(3)
        if self.theta != 0 or self.zeta != 0 or self.chi != 0:
            rx = protocols.unitary(cirq.rx(2 * self.theta))
            rz1 = protocols.unitary(cirq.rz(-self.zeta + self.chi))
            rz2 = protocols.unitary(cirq.rz(-self.zeta - self.chi))
            inner_matrix = rz1 @ rx @ rz2
            out = cirq.apply_matrix_to_slices(args.target_tensor, inner_matrix, slices=[oi, io], out=args.available_buffer)
        else:
            out = args.target_tensor
        if self.phi != 0:
            out[ii] *= cmath.exp(-1j * self.phi)
        if self.gamma != 0:
            f = cmath.exp(-1j * self.gamma)
            out[oi] *= f
            out[io] *= f
            out[ii] *= f * f
        return out

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        """Decomposes self into Z rotations and FSimGate.

        Note that Z rotations returned by this method have unusual global phase
        in that one of their eigenvalues is 1. This ensures the decomposition
        agrees with the matrix specified in class docstring. In particular, it
        makes the top left element of the matrix equal to 1.
        """

        def to_exponent(angle_rads: 'cirq.TParamVal') -> 'cirq.TParamVal':
            """Divides angle_rads by symbolic or numerical pi."""
            pi = sympy.pi if protocols.is_parameterized(angle_rads) else np.pi
            return angle_rads / pi
        q0, q1 = qubits
        before = self.rz_angles_before
        after = self.rz_angles_after
        yield (cirq.Z(q0) ** to_exponent(before[0]))
        yield (cirq.Z(q1) ** to_exponent(before[1]))
        yield FSimGate(self.theta, self.phi).on(q0, q1)
        yield (cirq.Z(q0) ** to_exponent(after[0]))
        yield (cirq.Z(q1) ** to_exponent(after[1]))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        theta = args.format_radians(self.theta)
        zeta = args.format_radians(self.zeta)
        chi = args.format_radians(self.chi)
        gamma = args.format_radians(self.gamma)
        phi = args.format_radians(self.phi)
        return (f'PhFSim({theta}, {zeta}, {chi}, {gamma}, {phi})', f'PhFSim({theta}, {zeta}, {chi}, {gamma}, {phi})')

    def __repr__(self) -> str:
        theta = proper_repr(self.theta)
        zeta = proper_repr(self.zeta)
        chi = proper_repr(self.chi)
        gamma = proper_repr(self.gamma)
        phi = proper_repr(self.phi)
        return f'cirq.PhasedFSimGate(theta={theta}, zeta={zeta}, chi={chi}, gamma={gamma}, phi={phi})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['theta', 'zeta', 'chi', 'gamma', 'phi'])

    def _num_qubits_(self) -> int:
        return 2