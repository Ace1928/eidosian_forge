import dataclasses
from typing import Any, Dict, Union
import cirq
@dataclasses.dataclass
class CalibrationLayer:
    """Python implementation of the proto found in
    cirq_google.api.v2.calibration_pb2.FocusedCalibrationLayer for use
    in Engine calls."""
    calibration_type: str
    program: cirq.Circuit
    args: Dict[str, Union[str, float]]

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['calibration_type', 'program', 'args'])