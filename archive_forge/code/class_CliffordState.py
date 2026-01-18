from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
@value.value_equality
class CliffordState:
    """A state of the Clifford simulation.

    The state is stored using Bravyi's CH-form which allows access to the full
    state vector (including phase).

    Gates and measurements are applied to each representation in O(n^2) time.
    """

    def __init__(self, qubit_map, initial_state: Union[int, 'cirq.StabilizerStateChForm']=0):
        self.qubit_map = qubit_map
        self.n = len(qubit_map)
        self.ch_form = initial_state if isinstance(initial_state, clifford.StabilizerStateChForm) else clifford.StabilizerStateChForm(self.n, initial_state)

    def _json_dict_(self):
        return {'qubit_map': [(k, v) for k, v in self.qubit_map.items()], 'ch_form': self.ch_form}

    @classmethod
    def _from_json_dict_(cls, qubit_map, ch_form, **kwargs):
        state = cls(dict(qubit_map))
        state.ch_form = ch_form
        return state

    def _value_equality_values_(self) -> Any:
        return (self.qubit_map, self.ch_form)

    def copy(self) -> 'cirq.CliffordState':
        state = CliffordState(self.qubit_map)
        state.ch_form = self.ch_form.copy()
        return state

    def __repr__(self) -> str:
        return repr(self.ch_form)

    def __str__(self) -> str:
        """Return the state vector string representation of the state."""
        return str(self.ch_form)

    def to_numpy(self) -> np.ndarray:
        return self.ch_form.to_state_vector()

    def state_vector(self):
        return self.ch_form.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        ch_form_args = clifford.StabilizerChFormSimulationState(prng=np.random.RandomState(), qubits=self.qubit_map.keys(), initial_state=self.ch_form)
        try:
            act_on(op, ch_form_args)
        except TypeError:
            raise ValueError(f'{op.gate} cannot be run with Clifford simulator.')
        return

    def apply_measurement(self, op: 'cirq.Operation', measurements: Dict[str, List[int]], prng: np.random.RandomState, collapse_state_vector=True):
        if not isinstance(op.gate, cirq.MeasurementGate):
            raise TypeError(f'apply_measurement only supports cirq.MeasurementGate operations. Found {op.gate} instead.')
        if collapse_state_vector:
            state = self
        else:
            state = self.copy()
        classical_data = value.ClassicalDataDictionaryStore()
        ch_form_args = clifford.StabilizerChFormSimulationState(prng=prng, classical_data=classical_data, qubits=self.qubit_map.keys(), initial_state=state.ch_form)
        act_on(op, ch_form_args)
        measurements.update({str(k): list(v[-1]) for k, v in classical_data.records.items()})