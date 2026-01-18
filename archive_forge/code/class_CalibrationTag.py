from typing import Any, Dict
import cirq
class CalibrationTag:
    """Tag to add onto an Operation that specifies alternate parameters.

    Google devices support the ability to run a procedure from calibration API
    that can tune the device for a specific circuit.  This will return a token
    as part of the result.  Attaching a `CalibrationTag` with that token
    specifies that the gate should use parameters from that specific
    calibration, instead of the default gate parameters.
    """

    def __init__(self, token: str):
        self.token = token

    def __str__(self) -> str:
        return f'CalibrationTag({self.token!r})'

    def __repr__(self) -> str:
        return f'cirq_google.CalibrationTag({self.token!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['token'])

    def __eq__(self, other) -> bool:
        if not isinstance(other, CalibrationTag):
            return NotImplemented
        return self.token == other.token

    def __hash__(self) -> int:
        return hash(self.token)