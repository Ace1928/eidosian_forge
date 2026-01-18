from typing import Any, cast, Optional, Type, Union
from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols
class ParallelGateFamily(gateset.GateFamily):
    """GateFamily which accepts instances of `cirq.ParallelGate` and its sub_gate.

    ParallelGateFamily is useful for description and validation of scenarios where multiple
    copies of a unitary gate can act in parallel. `cirq.ParallelGate` is used to express
    such a gate with a corresponding unitary `sub_gate` that acts in parallel.

    ParallelGateFamily supports initialization via:

    *    Gate Instances that can be applied in parallel.
    *    Gate Types whose instances can be applied in parallel.

    In both the cases, the users can specify an additional parameter `max_parallel_allowed` which
    is used to verify the maximum number of qubits on which any given gate instance can act on.

    To verify containment of a given `cirq.Gate` instance `g`, the gate family verfies that:

    *    `cirq.num_qubits(g)` <= `max_parallel_allowed` if `max_parallel_allowed` is not None.
    *    `g` or `g.sub_gate` (if `g` is an instance of `cirq.ParallelGate`) is an accepted gate
            based on type or instance checks depending on the initialization gate type.
    """

    def __init__(self, gate: Union[Type[raw_types.Gate], raw_types.Gate], *, name: Optional[str]=None, description: Optional[str]=None, max_parallel_allowed: Optional[int]=None) -> None:
        """Inits ParallelGateFamily

        Args:
            gate: The gate which can act in parallel. It can be a python `type` inheriting from
                `cirq.Gate` or a non-parameterized instance of a `cirq.Gate`. If an instance of
                `cirq.ParallelGate` is passed, then the corresponding `gate.sub_gate` is used.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            max_parallel_allowed: The maximum number of qubits on which a given gate `g`
            can act on. If None, then any number of qubits are allowed.
        """
        if isinstance(gate, parallel_gate.ParallelGate):
            if not max_parallel_allowed:
                max_parallel_allowed = protocols.num_qubits(gate)
            gate = gate.sub_gate
        self._max_parallel_allowed = max_parallel_allowed
        super().__init__(gate, name=name, description=description)

    def _max_parallel_str(self):
        return self._max_parallel_allowed if self._max_parallel_allowed is not None else 'INF'

    def _default_name(self) -> str:
        return f'{self._max_parallel_str()} Parallel ' + super()._default_name()

    def _default_description(self) -> str:
        check_type = 'g == {}' if isinstance(self.gate, raw_types.Gate) else 'isinstance(g, {})'
        return f'Accepts\n1. `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}` OR\n2. `cirq.ParallelGate` instance `g` s.t. `g.sub_gate` satisfies 1. and `cirq.num_qubits(g) <= {self._max_parallel_str()}` OR\n3. `cirq.Operation` instance `op` s.t. `op.gate` satisfies 1. or 2.'

    def _predicate(self, gate: raw_types.Gate) -> bool:
        if self._max_parallel_allowed is not None and protocols.num_qubits(gate) > self._max_parallel_allowed:
            return False
        gate = gate.sub_gate if isinstance(gate, parallel_gate.ParallelGate) else gate
        return super()._predicate(gate)

    def __repr__(self) -> str:
        name_and_description = ''
        if self.name != self._default_name() or self.description != self._default_description():
            name_and_description = f"""name="{self.name}", description=r'''{self.description}''', """
        return f'cirq.ParallelGateFamily(gate={self._gate_str(repr)}, {name_and_description}max_parallel_allowed={self._max_parallel_allowed})'

    def _value_equality_values_(self) -> Any:
        return super()._value_equality_values_() + (self._max_parallel_allowed,)

    def _json_dict_(self):
        return {'gate': self._gate_json(), 'name': self.name, 'description': self.description, 'max_parallel_allowed': self._max_parallel_allowed}

    @classmethod
    def _from_json_dict_(cls, gate, name, description, max_parallel_allowed, **kwargs):
        if isinstance(gate, str):
            gate = protocols.cirq_type_from_json(gate)
        return cls(gate, name=name, description=description, max_parallel_allowed=max_parallel_allowed)