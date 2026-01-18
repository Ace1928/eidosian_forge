from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
@value.value_equality
class _NoNoiseModel(NoiseModel):
    """A default noise model that adds no noise."""

    def noisy_moments(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']):
        return list(moments)

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.Moment':
        return moment

    def noisy_operation(self, operation: 'cirq.Operation') -> 'cirq.Operation':
        return operation

    def _value_equality_values_(self) -> Any:
        return None

    def __str__(self) -> str:
        return '(no noise)'

    def __repr__(self) -> str:
        return 'cirq.NO_NOISE'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [])

    def _has_unitary_(self) -> bool:
        return True

    def _has_mixture_(self) -> bool:
        return True