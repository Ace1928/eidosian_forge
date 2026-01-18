from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
@value.value_equality()
class Gateset:
    """Gatesets represent a collection of `cirq.GateFamily` objects.

    Gatesets are useful for

    - Describing the set of allowed gates in a human-readable format.
    - Validating a given gate / `cirq.OP_TREE` against the set of allowed gates.

    Gatesets rely on the underlying `cirq.GateFamily` for both description and
    validation purposes.
    """

    def __init__(self, *gates: Union[Type[raw_types.Gate], raw_types.Gate, GateFamily], name: Optional[str]=None, unroll_circuit_op: bool=True) -> None:
        """Init Gateset.

        Accepts a list of gates, each of which should be either

        - `cirq.Gate` subclass
        - `cirq.Gate` instance
        - `cirq.GateFamily` instance

        `cirq.Gate` subclasses and instances are converted to the default
        `cirq.GateFamily(gate=g)` instance and thus a default name and
        description is populated.

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
                `cirq.GateFamily` instances to initialize the Gateset.
            name: (Optional) Name for the Gateset. Useful for description.
            unroll_circuit_op: If True, `cirq.CircuitOperation` is recursively
                validated by validating the underlying `cirq.Circuit`.
        """
        self._name = name
        self._unroll_circuit_op = unroll_circuit_op
        self._instance_gate_families: Dict[raw_types.Gate, GateFamily] = {}
        self._type_gate_families: Dict[Type[raw_types.Gate], GateFamily] = {}
        self._gates_repr_str = ', '.join([_gate_str(g, repr) for g in gates])
        unique_gate_list: List[GateFamily] = list(dict.fromkeys((g if isinstance(g, GateFamily) else GateFamily(gate=g) for g in gates)))
        for g in unique_gate_list:
            if type(g) is GateFamily and (not (g.tags_to_ignore or g.tags_to_accept)):
                if isinstance(g.gate, raw_types.Gate):
                    self._instance_gate_families[g.gate] = g
                else:
                    self._type_gate_families[g.gate] = g
        self._unique_gate_list = unique_gate_list
        self._gates = frozenset(unique_gate_list)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def gates(self) -> FrozenSet[GateFamily]:
        return self._gates

    def with_params(self, *, name: Optional[str]=None, unroll_circuit_op: Optional[bool]=None) -> 'Gateset':
        """Returns a copy of this Gateset with identical gates and new values for named arguments.

        If a named argument is None then corresponding value of this Gateset is used instead.

        Args:
            name: New name for the Gateset.
            unroll_circuit_op: If True, new Gateset will recursively validate
                `cirq.CircuitOperation` by validating the underlying `cirq.Circuit`.

        Returns:
            `self` if all new values are None or identical to the values of current Gateset.
            else a new Gateset with identical gates and new values for named arguments.
        """

        def val_if_none(var: Any, val: Any) -> Any:
            return var if var is not None else val
        name = val_if_none(name, self._name)
        unroll_circuit_op = val_if_none(unroll_circuit_op, self._unroll_circuit_op)
        if name == self._name and unroll_circuit_op == self._unroll_circuit_op:
            return self
        gates = self.gates
        return Gateset(*gates, name=name, unroll_circuit_op=cast(bool, unroll_circuit_op))

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        """Check for containment of a given Gate/Operation in this Gateset.

        Containment checks are handled as follows:

        - For Gates or Operations that have an underlying gate (i.e. op.gate is not None):
            - Forwards the containment check to the underlying `cirq.GateFamily` objects.
            - Examples of such operations include `cirq.GateOperation`s and their controlled
                and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                `cirq.ControlledOperation` where `op.gate` is not None) etc.
        - For Operations that do not have an underlying gate:
            - Forwards the containment check to `self._validate_operation(item)`.
            - Examples of such operations include `cirq.CircuitOperation`s and their controlled
                and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                `cirq.ControlledOperation` where `op.gate` is None) etc.

        The complexity of the method in terms of the number of `gates`, n, is

        - O(1) when any default `cirq.GateFamily` instance accepts the given item, except
            for an Instance GateFamily trying to match an item with a different global phase.
        - O(n) for all other cases: matching against custom gate families, matching across
            global phase for the default Instance GateFamily, no match against any underlying
            gate family.

        Args:
            item: The `cirq.Gate` or `cirq.Operation` instance to check containment for.
        """
        if isinstance(item, raw_types.Operation) and item.gate is None:
            return self._validate_operation(item)
        g = item if isinstance(item, raw_types.Gate) else item.gate
        assert g is not None, f'`item`: {item} must be a gate or have a valid `item.gate`'
        if g in self._instance_gate_families:
            assert item in self._instance_gate_families[g], f'{item} instance matches {self._instance_gate_families[g]} but is not accepted by it.'
            return True
        for gate_mro_type in type(g).mro():
            if gate_mro_type in self._type_gate_families:
                assert item in self._type_gate_families[gate_mro_type], f'{g} type {gate_mro_type} matches Type GateFamily:{self._type_gate_families[gate_mro_type]} but is not accepted by it.'
                return True
        return any((item in gate_family for gate_family in self._gates))

    def validate(self, circuit_or_optree: Union['cirq.AbstractCircuit', op_tree.OP_TREE]) -> bool:
        """Validates gates forming `circuit_or_optree` should be contained in Gateset.

        Args:
            circuit_or_optree: The `cirq.Circuit` or `cirq.OP_TREE` to validate.
        """
        from cirq.circuits import circuit
        optree = circuit_or_optree
        if isinstance(circuit_or_optree, circuit.AbstractCircuit):
            optree = circuit_or_optree.all_operations()
        return all((self._validate_operation(op) for op in op_tree.flatten_to_ops(optree)))

    def _validate_operation(self, op: raw_types.Operation) -> bool:
        """Validates whether the given `cirq.Operation` is contained in this Gateset.

        The containment checks are handled as follows:

        - For any operation which has an underlying gate (i.e. `op.gate` is not None):
            - Containment is checked via `self.__contains__` which further checks for containment
                in any of the underlying gate families.
        - For all other types of operations (eg: `cirq.CircuitOperation`,
            etc):
            - The behavior is controlled via flags passed to the constructor.

        Users should override this method to define custom behavior for operations that do not
        have an underlying `cirq.Gate`.

        Args:
            op: The `cirq.Operation` instance to check containment for.
        """
        from cirq.circuits import circuit_operation
        if op.gate is not None:
            return op in self
        if isinstance(op, raw_types.TaggedOperation):
            return self._validate_operation(op.sub_operation)
        elif isinstance(op, circuit_operation.CircuitOperation) and self._unroll_circuit_op:
            return self.validate(op.mapped_circuit(deep=True))
        else:
            return False

    def _value_equality_values_(self) -> Any:
        return (self.gates, self.name, self._unroll_circuit_op)

    def __repr__(self) -> str:
        name_str = f'name = "{self.name}", ' if self.name is not None else ''
        gates_str = f'{self._gates_repr_str}, ' if len(self._gates_repr_str) > 0 else ''
        return f'cirq.Gateset({gates_str}{name_str}unroll_circuit_op = {self._unroll_circuit_op})'

    def __str__(self) -> str:
        header = 'Gateset: '
        if self.name:
            header += self.name
        return f'{header}\n' + '\n\n'.join([str(g) for g in self._unique_gate_list])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'gates': self._unique_gate_list, 'name': self.name, 'unroll_circuit_op': self._unroll_circuit_op}

    @classmethod
    def _from_json_dict_(cls, gates, name, unroll_circuit_op, **kwargs) -> 'Gateset':
        if 'accept_global_phase_op' in kwargs:
            accept_global_phase_op = kwargs['accept_global_phase_op']
            global_phase_family = GateFamily(gate=global_phase_op.GlobalPhaseGate)
            if accept_global_phase_op is True:
                gates.append(global_phase_family)
            elif accept_global_phase_op is False:
                gates = [family for family in gates if family.gate is not global_phase_op.GlobalPhaseGate]
        return cls(*gates, name=name, unroll_circuit_op=unroll_circuit_op)