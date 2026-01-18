from typing import DefaultDict, Dict, Sequence, TYPE_CHECKING, Optional
import abc
from collections import defaultdict
from cirq import circuits, devices, ops, protocols, transformers
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.permutation import (
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
class ExecutionStrategy(metaclass=abc.ABCMeta):
    """Tells `StrategyExecutorTransformer` how to execute an acquaintance strategy.

    An execution strategy tells `StrategyExecutorTransformer` how to execute
    an acquaintance strategy, i.e. what gates to implement at the available
    acquaintance opportunities."""
    keep_acquaintance = False

    @property
    @abc.abstractmethod
    def device(self) -> 'cirq.Device':
        """The device for which the executed acquaintance strategy should be
        valid.
        """

    @property
    @abc.abstractmethod
    def initial_mapping(self) -> LogicalMapping:
        """The initial mapping of logical indices to qubits."""

    @abc.abstractmethod
    def get_operations(self, indices: Sequence[LogicalIndex], qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Gets the logical operations to apply to qubits."""

    def __call__(self, *args, **kwargs):
        """Returns the final mapping of logical indices to qubits after
        executing an acquaintance strategy.
        """
        if len(args) < 1 or not isinstance(args[0], circuits.AbstractCircuit):
            raise ValueError('To call ExecutionStrategy, an argument of type circuits.AbstractCircuit must be passed in as the first non-keyword argument')
        input_circuit = args[0]
        strategy = StrategyExecutorTransformer(self)
        final_circuit = strategy(input_circuit, **kwargs)
        input_circuit._moments = final_circuit._moments
        return strategy.mapping