import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
@value.value_equality
class TaggedOperation(Operation):
    """Operation annotated with a set of tags.

    These Tags can be used for special processing.  TaggedOperations
    can be initialized with using `Operation.with_tags(tag)`
    or by using `TaggedOperation(op, tag)`.

    Tags added can be of any type, but they should be Hashable in order
    to allow equality checking.  If you wish to serialize operations into
    JSON, you should restrict yourself to only use objects that have a JSON
    serialization.

    See `Operation.with_tags()` for more information on intended usage.
    """

    def __init__(self, sub_operation: 'cirq.Operation', *tags: Hashable):
        self._sub_operation = sub_operation
        self._tags = tuple(tags)

    @property
    def sub_operation(self) -> 'cirq.Operation':
        return self._sub_operation

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self.sub_operation.qubits

    @property
    def gate(self) -> Optional['cirq.Gate']:
        return self.sub_operation.gate

    def with_qubits(self, *new_qubits: 'cirq.Qid'):
        return TaggedOperation(self.sub_operation.with_qubits(*new_qubits), *self._tags)

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        sub_op = protocols.with_measurement_key_mapping(self.sub_operation, key_map)
        if sub_op is NotImplemented:
            return NotImplemented
        return TaggedOperation(sub_op, *self.tags)

    def controlled_by(self, *control_qubits: 'cirq.Qid', control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None) -> 'cirq.Operation':
        if len(control_qubits) == 0:
            return self
        return self.sub_operation.controlled_by(*control_qubits, control_values=control_values)

    @property
    def tags(self) -> Tuple[Hashable, ...]:
        """Returns a tuple of the operation's tags."""
        return self._tags

    @property
    def untagged(self) -> 'cirq.Operation':
        """Returns the underlying operation without any tags."""
        return self.sub_operation

    def with_tags(self, *new_tags: Hashable) -> 'cirq.TaggedOperation':
        """Creates a new TaggedOperation with combined tags.

        Overloads Operation.with_tags to create a new TaggedOperation
        that has the tags of this operation combined with the new_tags
        specified as the parameter.
        """
        if not new_tags:
            return self
        return TaggedOperation(self.sub_operation, *self._tags, *new_tags)

    def __str__(self) -> str:
        tag_repr = ','.join((repr(t) for t in self._tags))
        return f'cirq.TaggedOperation({repr(self.sub_operation)}, {tag_repr})'

    def __repr__(self) -> str:
        return str(self)

    def _value_equality_values_(self) -> Any:
        return (self.sub_operation, self._tags)

    @classmethod
    def _from_json_dict_(cls, sub_operation, tags, **kwargs):
        return cls(sub_operation, *tags)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['sub_operation', 'tags'])

    def _decompose_(self) -> 'cirq.OP_TREE':
        return self._decompose_with_context_()

    def _decompose_with_context_(self, context: Optional['cirq.DecompositionContext']=None) -> 'cirq.OP_TREE':
        return protocols.decompose_once(self.sub_operation, default=None, flatten=False, context=context)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return protocols.pauli_expansion(self.sub_operation)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Union[np.ndarray, None, NotImplementedType]:
        return protocols.apply_unitary(self.sub_operation, args, default=None)

    @cached_method
    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_operation)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        return protocols.unitary(self.sub_operation, NotImplemented)

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
        return protocols.commutes(self.sub_operation, other, atol=atol)

    @cached_method
    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.sub_operation)

    def _mixture_(self) -> Sequence[Tuple[float, Any]]:
        return protocols.mixture(self.sub_operation, NotImplemented)

    @cached_method
    def _has_kraus_(self) -> bool:
        return protocols.has_kraus(self.sub_operation)

    def _kraus_(self) -> Union[Tuple[np.ndarray], NotImplementedType]:
        return protocols.kraus(self.sub_operation, NotImplemented)

    @cached_method
    def _measurement_key_names_(self) -> FrozenSet[str]:
        return protocols.measurement_key_names(self.sub_operation)

    @cached_method
    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        return protocols.measurement_key_objs(self.sub_operation)

    @cached_method
    def _is_measurement_(self) -> bool:
        sub = getattr(self.sub_operation, '_is_measurement_', None)
        if sub is not None:
            return sub()
        return NotImplemented

    @cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.sub_operation) or any((protocols.is_parameterized(tag) for tag in self.tags))

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> bool:
        sub = getattr(self.sub_operation, '_act_on_', None)
        if sub is not None:
            return sub(sim_state)
        return NotImplemented

    @cached_method
    def _parameter_names_(self) -> AbstractSet[str]:
        tag_params = {name for tag in self.tags for name in protocols.parameter_names(tag)}
        return protocols.parameter_names(self.sub_operation) | tag_params

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'TaggedOperation':
        resolved_op = protocols.resolve_parameters(self.sub_operation, resolver, recursive)
        resolved_tags = (protocols.resolve_parameters(tag, resolver, recursive) for tag in self._tags)
        return TaggedOperation(resolved_op, *resolved_tags)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        sub_op_info = protocols.circuit_diagram_info(self.sub_operation, args, NotImplemented)
        if sub_op_info is not NotImplemented and args.include_tags and sub_op_info.wire_symbols:
            sub_op_info.wire_symbols = (sub_op_info.wire_symbols[0] + str(list(self._tags)),) + sub_op_info.wire_symbols[1:]
        return sub_op_info

    @cached_method
    def _trace_distance_bound_(self) -> float:
        return protocols.trace_distance_bound(self.sub_operation)

    def _phase_by_(self, phase_turns: float, qubit_index: int) -> 'cirq.Operation':
        return protocols.phase_by(self.sub_operation, phase_turns, qubit_index)

    def __pow__(self, exponent: Any) -> 'cirq.Operation':
        return self.sub_operation ** exponent

    def __mul__(self, other: Any) -> Any:
        return self.sub_operation * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.sub_operation

    def _qasm_(self, args: 'protocols.QasmArgs') -> Optional[str]:
        return protocols.qasm(self.sub_operation, args=args, default=None)

    def _equal_up_to_global_phase_(self, other: Any, atol: Union[int, float]=1e-08) -> Union[NotImplementedType, bool]:
        return protocols.equal_up_to_global_phase(self.sub_operation, other, atol=atol)

    @property
    def classical_controls(self) -> FrozenSet['cirq.Condition']:
        return self.sub_operation.classical_controls

    def without_classical_controls(self) -> 'cirq.Operation':
        new_sub_operation = self.sub_operation.without_classical_controls()
        return self if new_sub_operation is self.sub_operation else new_sub_operation

    def with_classical_controls(self, *conditions: Union[str, 'cirq.MeasurementKey', 'cirq.Condition', sympy.Expr]) -> 'cirq.Operation':
        if not conditions:
            return self
        return self.sub_operation.with_classical_controls(*conditions)

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        return protocols.control_keys(self.sub_operation)