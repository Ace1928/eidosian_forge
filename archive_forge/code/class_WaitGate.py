from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union
import sympy
from cirq import value, protocols
from cirq.ops import raw_types
@value.value_equality
class WaitGate(raw_types.Gate):
    """An idle gate that represents waiting.

    In non-noisy simulators, this gate is just an identity gate. But noisy
    simulators and noise models may insert more error for longer waits.
    """

    def __init__(self, duration: 'cirq.DURATION_LIKE', num_qubits: Optional[int]=None, qid_shape: Optional[Tuple[int, ...]]=None) -> None:
        """Initialize a wait gate with the given duration.

        Args:
            duration: A constant or parameterized wait duration. This can be
                an instance of `datetime.timedelta` or `cirq.Duration`.
            num_qubits: The number of qubits the gate operates on. If None and `qid_shape` is None,
                this defaults to one qubit.
            qid_shape: Can be specified instead of `num_qubits` for the case that the gate should
                act on qudits.

        Raises:
            ValueError: If the `qid_shape` provided is empty or `num_qubits` contradicts
                `qid_shape`.
        """
        self._duration = value.Duration(duration)
        if not protocols.is_parameterized(self.duration) and self.duration < 0:
            raise ValueError('duration < 0')
        if qid_shape is None:
            if num_qubits is None:
                qid_shape = (2,)
            else:
                qid_shape = (2,) * num_qubits
        if num_qubits is None:
            num_qubits = len(qid_shape)
        if not qid_shape:
            raise ValueError('Waiting on an empty set of qubits.')
        if num_qubits != len(qid_shape):
            raise ValueError('len(qid_shape) != num_qubits')
        self._qid_shape = qid_shape

    @property
    def duration(self) -> 'cirq.Duration':
        return self._duration

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.duration)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.duration)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'WaitGate':
        return WaitGate(protocols.resolve_parameters(self.duration, resolver, recursive), qid_shape=self._qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args):
        return args.target_tensor

    def _decompose_(self, qubits):
        return []

    def _trace_distance_bound_(self):
        return 0

    def __pow__(self, power):
        if power == 1 or power == -1:
            return self
        return NotImplemented

    def __str__(self) -> str:
        return f'WaitGate({self.duration})'

    def __repr__(self) -> str:
        return f'cirq.WaitGate({repr(self.duration)})'

    def _json_dict_(self) -> Dict[str, Any]:
        d = protocols.obj_to_dict_helper(self, ['duration'])
        if len(self._qid_shape) != 1:
            d['num_qubits'] = len(self._qid_shape)
        if any((d != 2 for d in self._qid_shape)):
            d['qid_shape'] = self._qid_shape
        return d

    @classmethod
    def _from_json_dict_(cls, duration, num_qubits=None, qid_shape=None, **kwargs):
        return cls(duration=duration, num_qubits=num_qubits, qid_shape=None if qid_shape is None else tuple(qid_shape))

    def _value_equality_values_(self) -> Any:
        return self.duration