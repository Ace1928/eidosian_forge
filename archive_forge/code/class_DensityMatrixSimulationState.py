from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
class DensityMatrixSimulationState(SimulationState[_BufferedDensityMatrix]):
    """State and context for an operation acting on a density matrix.

    To act on this object, directly edit the `target_tensor` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
    """

    def __init__(self, *, available_buffer: Optional[List[np.ndarray]]=None, prng: Optional[np.random.RandomState]=None, qubits: Optional[Sequence['cirq.Qid']]=None, initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE']=0, dtype: Type[np.complexfloating]=np.complex64, classical_data: Optional['cirq.ClassicalDataStore']=None):
        """Inits DensityMatrixSimulationState.

        Args:
            available_buffer: A workspace with the same shape and dtype as
                `target_tensor`. Used by operations that cannot be applied to
                `target_tensor` inline, in order to avoid unnecessary
                allocations.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            initial_state: The initial state for the simulation in the
                computational basis.
            dtype: The `numpy.dtype` of the inferred state vector. One of
                `numpy.complex64` or `numpy.complex128`. Only used when
                `target_tenson` is None.
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: If `initial_state` is provided as integer, but `qubits`
                is not provided.
        """
        state = _BufferedDensityMatrix.create(initial_state=initial_state, qid_shape=tuple((q.dimension for q in qubits)) if qubits is not None else None, dtype=dtype, buffer=available_buffer)
        super().__init__(state=state, prng=prng, qubits=qubits, classical_data=classical_data)

    def add_qubits(self, qubits: Sequence['cirq.Qid']):
        ret = super().add_qubits(qubits)
        return self.kronecker_product(type(self)(qubits=qubits), inplace=True) if ret is NotImplemented else ret

    def remove_qubits(self, qubits: Sequence['cirq.Qid']):
        ret = super().remove_qubits(qubits)
        if ret is not NotImplemented:
            return ret
        extracted, remainder = self.factor(qubits, inplace=True)
        remainder._state._density_matrix *= extracted._state._density_matrix.reshape(-1)[0]
        return remainder

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool=True) -> bool:
        strats: List[Callable[[Any, Any, Sequence['cirq.Qid']], bool]] = [_strat_apply_channel_to_state]
        if allow_decompose:
            strats.append(strat_act_on_from_apply_decompose)
        for strat in strats:
            result = strat(action, self, qubits)
            if result is False:
                break
            if result is True:
                return True
            assert result is NotImplemented, str(result)
        raise TypeError(f"Can't simulate operations that don't implement SupportsUnitary, SupportsConsistentApplyUnitary, SupportsMixture or SupportsKraus or is a measurement: {action!r}")

    def __repr__(self) -> str:
        return f'cirq.DensityMatrixSimulationState(initial_state={proper_repr(self.target_tensor)}, qubits={self.qubits!r}, classical_data={self.classical_data!r})'

    @property
    def target_tensor(self):
        return self._state._density_matrix

    @property
    def available_buffer(self):
        return self._state._buffer

    @property
    def qid_shape(self):
        return self._state._qid_shape