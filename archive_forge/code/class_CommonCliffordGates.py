from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
class CommonCliffordGates(metaclass=CommonCliffordGateMetaClass):

    @classmethod
    def from_clifford_tableau(cls, tableau: qis.CliffordTableau) -> 'CliffordGate':
        """Create the CliffordGate instance from Clifford Tableau.

        Args:
            tableau: A CliffordTableau to define the effect of Clifford Gate applying on
            the stabilizer state or Pauli group. The meaning of tableau here is
                    To  X   Z    sign
            from  X  [ X_x Z_x | r_x ]
            from  Z  [ X_z Z_z | r_z ]
            Each row in the Clifford tableau indicates how the transformation of original
            Pauli gates to the new gates after applying this Clifford Gate.

        Returns:
            A CliffordGate instance, which has the transformation defined by
            the input tableau.

        Raises:
            ValueError: When input tableau is wrong type or the tableau does not
            satisfy the symplectic property.
        """
        if not isinstance(tableau, qis.CliffordTableau):
            raise ValueError('Input argument has to be a CliffordTableau instance.')
        if not tableau._validate():
            raise ValueError('It is not a valid Clifford tableau.')
        return CliffordGate(_clifford_tableau=tableau)

    @classmethod
    def from_op_list(cls, operations: Sequence[raw_types.Operation], qubit_order: Sequence[raw_types.Qid]) -> 'CliffordGate':
        """Construct a new Clifford gates from several known operations.

        Args:
            operations: A list of cirq operations to construct the Clifford gate.
                The combination order is the first element in the list applies the transformation
                on the stabilizer state first.
            qubit_order: Determines how qubits are ordered when decomposite the operations.

        Returns:
            A CliffordGate instance, which has the transformation on the stabilizer
            state equivalent to the composition of operations.

        Raises:
            ValueError: When one or more operations do not have stabilizer effect.
        """
        for op in operations:
            if op.gate and op.gate._has_stabilizer_effect_():
                continue
            raise ValueError('Clifford Gate can only be constructed from the operations that has stabilizer effect.')
        base_tableau = qis.CliffordTableau(len(qubit_order))
        args = sim.clifford.CliffordTableauSimulationState(tableau=base_tableau, qubits=qubit_order, prng=np.random.RandomState(0))
        for op in operations:
            protocols.act_on(op, args, allow_decompose=True)
        return CliffordGate.from_clifford_tableau(args.tableau)

    @classmethod
    def _from_json_dict_(cls, n, rs, xs, zs, **kwargs):
        _clifford_tableau = qis.CliffordTableau._from_json_dict_(n, rs, xs, zs)
        return cls(_clifford_tableau=_clifford_tableau)

    @classmethod
    def _get_sqrt_map(cls) -> Dict[float, Dict['SingleQubitCliffordGate', 'SingleQubitCliffordGate']]:
        """Returns a map containing two keys 0.5 and -0.5 for the sqrt mapping of Pauli gates."""
        return {0.5: {cls.X: cls.X_sqrt, cls.Y: cls.Y_sqrt, cls.Z: cls.Z_sqrt}, -0.5: {cls.X: cls.X_nsqrt, cls.Y: cls.Y_nsqrt, cls.Z: cls.Z_nsqrt}}