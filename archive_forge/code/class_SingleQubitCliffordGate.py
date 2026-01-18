from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@value.value_equality(manual_cls=True)
class SingleQubitCliffordGate(CliffordGate):
    """Any single qubit Clifford rotation."""

    def __init__(self, *, _clifford_tableau: qis.CliffordTableau) -> None:
        super().__init__(_clifford_tableau=_clifford_tableau)

    def _num_qubits_(self):
        return 1

    @staticmethod
    def from_clifford_tableau(tableau: qis.CliffordTableau) -> 'SingleQubitCliffordGate':
        if not isinstance(tableau, qis.CliffordTableau):
            raise ValueError('Input argument has to be a CliffordTableau instance.')
        if not tableau._validate():
            raise ValueError('Input tableau is not a valid Clifford tableau.')
        if tableau.n != 1:
            raise ValueError('The number of qubit of input tableau should be 1 for SingleQubitCliffordGate.')
        return SingleQubitCliffordGate(_clifford_tableau=tableau)

    @staticmethod
    def from_xz_map(x_to: Tuple[Pauli, bool], z_to: Tuple[Pauli, bool]) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the specified transforms.
        The Y transform is derived from the X and Z.

        Args:
            x_to: Which Pauli to transform X to and if it should negate.
            z_to: Which Pauli to transform Z to and if it should negate.
        """
        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))

    @staticmethod
    def from_single_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]=None, *, x_to: Optional[Tuple[Pauli, bool]]=None, y_to: Optional[Tuple[Pauli, bool]]=None, z_to: Optional[Tuple[Pauli, bool]]=None) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        The arguments are exclusive, only one may be specified.

        Args:
            pauli_map_to: A dictionary with a single key value pair describing
                the transform.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(1, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        (trans_from, (trans_to, flip)), = tuple(rotation_map.items())
        if trans_from == trans_to:
            trans_from2 = Pauli.by_relative_index(trans_to, 1)
            trans_to2 = Pauli.by_relative_index(trans_from, 1)
            flip2 = False
        else:
            trans_from2 = trans_to
            trans_to2 = trans_from
            flip2 = not flip
        rotation_map[trans_from2] = (trans_to2, flip2)
        return SingleQubitCliffordGate.from_double_map(rotation_map)

    @staticmethod
    def from_double_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]=None, *, x_to: Optional[Tuple[Pauli, bool]]=None, y_to: Optional[Tuple[Pauli, bool]]=None, z_to: Optional[Tuple[Pauli, bool]]=None) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

        Args:
            pauli_map_to: A dictionary with two key value pairs describing
                two transforms.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(2, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        (from1, trans1), (from2, trans2) = tuple(rotation_map.items())
        from3 = from1.third(from2)
        to3 = trans1[0].third(trans2[0])
        flip3 = trans1[1] ^ trans2[1] ^ ((from1 < from2) != (trans1[0] < trans2[0]))
        rotation_map[from3] = (to3, flip3)
        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))

    @staticmethod
    def from_pauli(pauli: Pauli, sqrt: bool=False) -> 'SingleQubitCliffordGate':
        prev_pauli = Pauli.by_relative_index(pauli, -1)
        next_pauli = Pauli.by_relative_index(pauli, 1)
        if sqrt:
            rotation_map = {prev_pauli: (next_pauli, True), pauli: (pauli, False), next_pauli: (prev_pauli, False)}
        else:
            rotation_map = {prev_pauli: (prev_pauli, True), pauli: (pauli, False), next_pauli: (next_pauli, True)}
        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))

    @staticmethod
    def from_quarter_turns(pauli: Pauli, quarter_turns: int) -> 'SingleQubitCliffordGate':
        quarter_turns = quarter_turns % 4
        if quarter_turns == 0:
            return SingleQubitCliffordGate.I
        if quarter_turns == 1:
            return SingleQubitCliffordGate.from_pauli(pauli, True)
        if quarter_turns == 2:
            return SingleQubitCliffordGate.from_pauli(pauli)
        return SingleQubitCliffordGate.from_pauli(pauli, True) ** (-1)

    @staticmethod
    def from_unitary(u: np.ndarray) -> Optional['SingleQubitCliffordGate']:
        """Creates Clifford gate with given unitary (up to global phase).

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            SingleQubitCliffordGate, whose matrix is equal to given matrix (up
            to global phase), or `None` if `u` is not a matrix of a single-qubit
            Clifford gate.
        """
        if u.shape != (2, 2) or not linalg.is_unitary(u):
            return None
        x = protocols.unitary(pauli_gates.X)
        z = protocols.unitary(pauli_gates.Z)
        x_to = _to_pauli_tuple(u @ x @ u.conj().T)
        z_to = _to_pauli_tuple(u @ z @ u.conj().T)
        if x_to is None or z_to is None:
            return None
        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))

    @classmethod
    def from_unitary_with_global_phase(cls, u: np.ndarray) -> Optional[Tuple['SingleQubitCliffordGate', complex]]:
        """Creates Clifford gate with given unitary, including global phase.

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            A tuple of a SingleQubitCliffordGate and a global phase, such that
            the gate unitary (as given by `cirq.unitary`) times the global phase
            is identical to the given unitary `u`; or `None` if `u` is not the
            matrix of a single-qubit Clifford gate.
        """
        gate = cls.from_unitary(u)
        if gate is None:
            return None
        k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
        return (gate, u[k] / protocols.unitary(gate)[k])

    def pauli_tuple(self, pauli: Pauli) -> Tuple[Pauli, bool]:
        """Returns a tuple of a Pauli operator and a boolean.

        The pauli is the operator of the transform and the boolean
        determines whether the operator should be flipped.  For instance,
        it is True if the coefficient is -1, and False if the coefficient
        is 1.
        """
        x_to = self._clifford_tableau.destabilizers()[0]
        z_to = self._clifford_tableau.stabilizers()[0]
        if pauli == pauli_gates.X:
            to = x_to
        elif pauli == pauli_gates.Z:
            to = z_to
        else:
            to = x_to * z_to
            to._coefficient *= 1j
        to_gate = Pauli._XYZ[to.pauli_mask[0] - 1]
        return (to_gate, bool(to.coefficient != 1.0))

    def dense_pauli_string(self, pauli: Pauli) -> 'cirq.DensePauliString':
        from cirq.ops import dense_pauli_string
        pauli_tuple = self.pauli_tuple(pauli)
        coefficient = -1 if pauli_tuple[1] else 1
        return dense_pauli_string.DensePauliString(str(pauli_tuple[0]), coefficient=coefficient)

    def to_phased_xz_gate(self) -> phased_x_z_gate.PhasedXZGate:
        """Convert this gate to a PhasedXZGate instance.

        The rotation can be categorized by {axis} * {degree}:
            * Identity: I
            * {x, y, z} * {90, 180, 270}  --- {X, Y, Z} + 6 Quarter turn gates
            * {+/-xy, +/-yz, +/-zx} * 180  --- 6 Hadamard-like gates
            * {middle point of xyz in 4 Quadrant} * {120, 240} --- swapping axis
        note 1 + 9 + 6 + 8 = 24 in total.

        To associate with Clifford Tableau, it can also be grouped by 4:
            * {I,X,Y,Z} is [[1 0], [0, 1]]
            * {+/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane} is [[1, 0], [1, 1]]
            * {+/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane} is [[1, 1], [0, 1]]
            * {+/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane} is [[0, 1], [1, 0]]
            * {middle point of xyz in 4 Quadrant} * 120 is [[0, 1], [1, 1]]
            * {middle point of xyz in 4 Quadrant} * 240 is [[1, 1], [1, 0]]
        """
        x_to_flip, z_to_flip = self.clifford_tableau.rs
        flip_index = int(z_to_flip) * 2 + x_to_flip
        a, x, z = (0.0, 0.0, 0.0)
        matrix = self.clifford_tableau.matrix()
        if np.array_equal(matrix, [[1, 0], [0, 1]]):
            to_phased_xz = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif np.array_equal(matrix, [[1, 0], [1, 1]]):
            a = 0.0
            x = 0.5 if x_to_flip ^ z_to_flip else -0.5
            z = 1.0 if x_to_flip else 0.0
        elif np.array_equal(matrix, [[0, 1], [1, 0]]):
            a = 0.5
            x = 0.5 if x_to_flip else -0.5
            z = 0.0 if x_to_flip ^ z_to_flip else 1.0
        elif np.array_equal(matrix, [[1, 1], [0, 1]]):
            to_phased_xz = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.25, 1.0, 0.0), (-0.25, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif np.array_equal(matrix, [[0, 1], [1, 1]]):
            a = 0.5
            x = 0.5 if x_to_flip else -0.5
            z = 0.5 if x_to_flip ^ z_to_flip else -0.5
        else:
            assert np.array_equal(matrix, [[1, 1], [1, 0]])
            a = 0.0
            x = -0.5 if x_to_flip ^ z_to_flip else 0.5
            z = -0.5 if x_to_flip else 0.5
        return phased_x_z_gate.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    def __pow__(self, exponent) -> 'SingleQubitCliffordGate':
        if self._get_sqrt_map().get(exponent, None):
            pow_gate = self._get_sqrt_map()[exponent].get(self, None)
            if pow_gate:
                return pow_gate
        ret_gate = super().__pow__(exponent)
        if ret_gate is NotImplemented:
            return NotImplemented
        return SingleQubitCliffordGate.from_clifford_tableau(ret_gate.clifford_tableau)

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']):
        return NotImplemented

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit, = qubits
        return tuple((gate.on(qubit) for gate in self.decompose_gate()))

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
        if isinstance(other, SingleQubitCliffordGate):
            return self.commutes_with_single_qubit_gate(other)
        if isinstance(other, Pauli):
            return self.commutes_with_pauli(other)
        return NotImplemented

    def commutes_with_single_qubit_gate(self, gate: 'SingleQubitCliffordGate') -> bool:
        """Tests if the two circuits would be equivalent up to global phase:
        --self--gate-- and --gate--self--"""
        self_then_gate = self.clifford_tableau.then(gate.clifford_tableau)
        gate_then_self = gate.clifford_tableau.then(self.clifford_tableau)
        return self_then_gate == gate_then_self

    def commutes_with_pauli(self, pauli: Pauli) -> bool:
        to, flip = self.pauli_tuple(pauli)
        return to == pauli and (not flip)

    def merged_with(self, second: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
        return SingleQubitCliffordGate.from_clifford_tableau(self.clifford_tableau.then(second.clifford_tableau))

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        mat = np.eye(2)
        qubit = named_qubit.NamedQubit('arbitrary')
        for op in protocols.decompose_once_with_qubits(self, (qubit,)):
            mat = protocols.unitary(op).dot(mat)
        return mat

    def decompose_gate(self) -> Sequence['cirq.Gate']:
        """Decomposes this clifford into a series of H and pauli rotation gates.

        Returns:
            A sequence of H and pauli rotation gates which are equivalent to this
            clifford gate if applied in order. This decomposition agrees with
            cirq.unitary(self), including global phase.
        """
        if self == SingleQubitCliffordGate.H:
            return [common_gates.H]
        rotations = self.decompose_rotation()
        return [r ** (qt / 2) for r, qt in rotations]

    def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
        """Decomposes this clifford into a series of pauli rotations.

        Each rotation is given as a tuple of (axis, quarter_turns),
        where axis is a Pauli giving the axis to rotate about. The
        result will be a sequence of zero, one, or two rotations.

        Note that the combined unitary effect of these rotations may
        differ from cirq.unitary(self) by a global phase.
        """
        x_rot = self.pauli_tuple(pauli_gates.X)
        y_rot = self.pauli_tuple(pauli_gates.Y)
        z_rot = self.pauli_tuple(pauli_gates.Z)
        whole_arr = (x_rot[0] == pauli_gates.X, y_rot[0] == pauli_gates.Y, z_rot[0] == pauli_gates.Z)
        num_whole = sum(whole_arr)
        flip_arr = (x_rot[1], y_rot[1], z_rot[1])
        num_flip = sum(flip_arr)
        if num_whole == 3:
            if num_flip == 0:
                return []
            pauli = Pauli.by_index(flip_arr.index(False))
            return [(pauli, 2)]
        if num_whole == 1:
            index = whole_arr.index(True)
            pauli = Pauli.by_index(index)
            next_pauli = Pauli.by_index(index + 1)
            flip = flip_arr[index]
            output = []
            if flip:
                output.append((next_pauli, 2))
            if self.pauli_tuple(next_pauli)[1]:
                output.append((pauli, -1))
            else:
                output.append((pauli, 1))
            return output
        elif num_whole == 0:
            if x_rot[0] == pauli_gates.Y:
                return [(pauli_gates.X, -1 if y_rot[1] else 1), (pauli_gates.Z, -1 if x_rot[1] else 1)]
            return [(pauli_gates.Z, 1 if y_rot[1] else -1), (pauli_gates.X, 1 if z_rot[1] else -1)]
        assert False, 'Impossible condition where this gate only rotates one Pauli to a different Pauli.'

    def equivalent_gate_before(self, after: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
        return self.merged_with(after).merged_with(self ** (-1))

    def __repr__(self) -> str:
        return f'cirq.CliffordGate.from_clifford_tableau({self.clifford_tableau!r})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        well_known_map = {SingleQubitCliffordGate.I: 'I', SingleQubitCliffordGate.H: 'H', SingleQubitCliffordGate.X: 'X', SingleQubitCliffordGate.Y: 'Y', SingleQubitCliffordGate.Z: 'Z', SingleQubitCliffordGate.X_sqrt: 'X', SingleQubitCliffordGate.Y_sqrt: 'Y', SingleQubitCliffordGate.Z_sqrt: 'Z', SingleQubitCliffordGate.X_nsqrt: 'X', SingleQubitCliffordGate.Y_nsqrt: 'Y', SingleQubitCliffordGate.Z_nsqrt: 'Z'}
        if self in well_known_map:
            symbol = well_known_map[self]
        else:
            rotations = self.decompose_rotation()
            symbol = '-'.join((str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2) for r, qt in rotations))
            symbol = f'({symbol})'
        return protocols.CircuitDiagramInfo(wire_symbols=(symbol,), exponent={SingleQubitCliffordGate.X_sqrt: 0.5, SingleQubitCliffordGate.Y_sqrt: 0.5, SingleQubitCliffordGate.Z_sqrt: 0.5, SingleQubitCliffordGate.X_nsqrt: -0.5, SingleQubitCliffordGate.Y_nsqrt: -0.5, SingleQubitCliffordGate.Z_nsqrt: -0.5}.get(self, 1))

    def _value_equality_values_(self):
        return self._clifford_tableau.matrix().tobytes() + self._clifford_tableau.rs.tobytes()

    def _value_equality_values_cls_(self):
        """To make it with compatible to compare with clifford gate."""
        return CliffordGate