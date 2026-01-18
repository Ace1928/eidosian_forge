from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
class CompilationTargetGateset(ops.Gateset, metaclass=abc.ABCMeta):
    """Abstract base class to create gatesets that can be used as targets for compilation.

    An instance of this type can be passed to transformers like `cirq.convert_to_target_gateset`,
    which can transform any given circuit to contain gates accepted by this gateset.
    """

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Maximum number of qubits on which a gate from this gateset can act upon."""

    @abc.abstractmethod
    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Method to rewrite the given operation using gates from this gateset.

        Args:
            op: `cirq.Operation` to be rewritten using gates from this gateset.
            moment_idx: Moment index where the given operation `op` occurs in a circuit.

        Returns:
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from this gateset.
            - `None` or `NotImplemented` if does not know how to decompose `op`.
        """

    def _validate_operation(self, op: 'cirq.Operation') -> bool:
        """Validates whether the given `cirq.Operation` is contained in this Gateset.

        Overrides the method on the base gateset class to ensure that operations which created
        as intermediate compilation results are not accepted.
        For example, if a preprocessing `merge_k_qubit_unitaries` transformer merges connected
        component of 2q unitaries, it should not be accepted in the gateset so that so we can
        use `decompose_to_target_gateset` to determine how to expand this component.

        Args:
            op: The `cirq.Operation` instance to check containment for.

        Returns:
            Whether the given operation is contained in the gateset.
        """
        if self._intermediate_result_tag in op.tags:
            return False
        return super()._validate_operation(op)

    @property
    def _intermediate_result_tag(self) -> Hashable:
        """A tag used to identify intermediate compilation results."""
        return '_default_merged_k_qubit_unitaries'

    @property
    def preprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run before decomposing individual operations."""
        return [create_transformer_with_kwargs(transformers.expand_composite, no_decomp=lambda op: protocols.num_qubits(op) <= self.num_qubits), create_transformer_with_kwargs(merge_k_qubit_gates.merge_k_qubit_unitaries, k=self.num_qubits, rewriter=lambda op: op.with_tags(self._intermediate_result_tag))]

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [merge_single_qubit_gates.merge_single_qubit_moments_to_phxz, transformers.drop_negligible_operations, transformers.drop_empty_moments]