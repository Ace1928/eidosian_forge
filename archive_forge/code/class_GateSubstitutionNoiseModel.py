from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
class GateSubstitutionNoiseModel(NoiseModel):

    def __init__(self, substitution_func: Callable[['cirq.Operation'], 'cirq.Operation']):
        """Noise model which replaces operations using a substitution function.

        Args:
            substitution_func: a function for replacing operations.
        """
        self.substitution_func = substitution_func

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return moment_module.Moment([self.substitution_func(op) for op in moment.operations])