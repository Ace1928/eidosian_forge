from typing import Any, cast, Optional, Type, Union
from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols
class AnyIntegerPowerGateFamily(gateset.GateFamily):
    """GateFamily which accepts instances of a given `cirq.EigenGate`, raised to integer power."""

    def __init__(self, gate: Type[eigen_gate.EigenGate]) -> None:
        """Init AnyIntegerPowerGateFamily

        Args:
            gate: A subclass of `cirq.EigenGate` s.t. an instance `g` of `gate` will be
                accepted if `g.exponent` is an integer.

        Raises:
            ValueError: If `gate` is not a subclass of `cirq.EigenGate`.
        """
        if not (isinstance(gate, type) and issubclass(gate, eigen_gate.EigenGate)):
            raise ValueError(f'{gate} must be a subclass of `cirq.EigenGate`.')
        super().__init__(gate, name=f'AnyIntegerPowerGateFamily: {gate}', description=f'Accepts any instance `g` of `{gate}` s.t. `g.exponent` is an integer.')

    def _predicate(self, g: raw_types.Gate) -> bool:
        if protocols.is_parameterized(g) or not super()._predicate(g):
            return False
        exp = cast(eigen_gate.EigenGate, g).exponent
        return int(exp) == exp

    def __repr__(self) -> str:
        return f'cirq.AnyIntegerPowerGateFamily({self._gate_str()})'

    def _value_equality_values_(self) -> Any:
        return self.gate

    def _json_dict_(self):
        return {'gate': self._gate_json()}

    @classmethod
    def _from_json_dict_(cls, gate, **kwargs):
        if isinstance(gate, str):
            gate = protocols.cirq_type_from_json(gate)
        return cls(gate)