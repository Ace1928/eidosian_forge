from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
def _noisy_operation_impl_moment(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
    return self.noisy_moment(moment_module.Moment([operation]), operation.qubits)