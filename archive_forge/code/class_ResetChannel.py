import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@value.value_equality
class ResetChannel(raw_types.Gate):
    """Reset a qubit back to its |0⟩ state.

    The reset channel is equivalent to performing an unobserved measurement
    which then controls a bit flip onto the targeted qubit.

    This channel evolves a density matrix as follows:

    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger + M_1 \\rho M_1^\\dagger
    $$

    With:

    $$
    \\begin{aligned}
    M_0 =& \\begin{bmatrix}
            1 & 0  \\\\
            0 & 0
          \\end{bmatrix}
    \\\\
    M_1 =& \\begin{bmatrix}
            0 & 1 \\\\
            0 & 0
          \\end{bmatrix}
    \\end{aligned}
    $$
    """

    def __init__(self, dimension: int=2) -> None:
        """Construct channel that resets to the zero state.

        Args:
            dimension: Specify this argument when resetting a qudit.  There will
                be `dimension` number of dimension by dimension matrices
                describing the channel, each with a 1 at a different position in
                the top row.
        """
        self._dimension = dimension

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('reset {0};\n', qubits[0])

    def _qid_shape_(self):
        return (self._dimension,)

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']):
        if len(qubits) != 1:
            return NotImplemented
        from cirq.sim import simulation_state
        if isinstance(sim_state, simulation_state.SimulationState) and (not sim_state.can_represent_mixed_states):
            result = sim_state._perform_measurement(qubits)[0]
            gate = common_gates.XPowGate(dimension=self.dimension) ** (self.dimension - result)
            protocols.act_on(gate, sim_state, qubits)
            return True
        return NotImplemented

    def _kraus_(self) -> Iterable[np.ndarray]:
        channel = np.zeros((self._dimension,) * 3, dtype=np.complex64)
        channel[:, 0, :] = np.eye(self._dimension)
        return channel

    def _apply_channel_(self, args: 'cirq.ApplyChannelArgs'):
        configs = []
        for i in range(self._dimension):
            s1 = transformations._SliceConfig(axis=args.left_axes[0], source_index=i, target_index=0)
            s2 = transformations._SliceConfig(axis=args.right_axes[0], source_index=i, target_index=0)
            configs.append(transformations._BuildFromSlicesArgs(slices=(s1, s2), scale=1))
        transformations._build_from_slices(configs, args.target_tensor, out=args.out_buffer)
        return args.out_buffer

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._dimension

    def __repr__(self) -> str:
        if self._dimension == 2:
            return 'cirq.ResetChannel()'
        else:
            return f'cirq.ResetChannel(dimension={self._dimension!r})'

    def __str__(self) -> str:
        return 'reset'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        return 'R'

    @property
    def dimension(self) -> int:
        """The dimension of the qudit being reset."""
        return self._dimension

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['dimension'])