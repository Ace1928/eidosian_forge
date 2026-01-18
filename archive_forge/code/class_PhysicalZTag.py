from typing import Any, Dict
import cirq
class PhysicalZTag:
    """Class to add as a tag onto an Operation to denote a Physical Z operation.

    By default, all Z rotations on Google devices are considered to be virtual.
    When performing the Z operation, the device will update its internal phase
    tracking mechanisms, essentially commuting it forwards through the circuit
    until it hits a non-commuting operation (Such as a sqrt(iSwap)).

    When applied to a Z rotation operation, this tag indicates to the hardware
    that an actual physical operation should be done instead.  This class can
    only be applied to instances of `cirq.ZPowGate`.  If applied to other gates
    (such as PhasedXZGate), this class will have no effect.
    """

    def __str__(self) -> str:
        return 'PhysicalZTag()'

    def __repr__(self) -> str:
        return 'cirq_google.PhysicalZTag()'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalZTag)

    def __hash__(self) -> int:
        return 123