from typing import Any, Dict
from cirq import protocols
from cirq.ops import raw_types
class NoIdentifierQubit(raw_types.Qid):
    """A singleton qubit type that does not have a qudit variant.
    This is useful for testing code that wraps qubits as qudits.
    """

    def __init__(self) -> None:
        pass

    def _comparison_key(self):
        return ()

    @property
    def dimension(self) -> int:
        return 2

    def __repr__(self) -> str:
        return 'cirq.testing.NoIdentifierQubit()'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [])