import dataclasses
import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING
@dataclasses.dataclass
class CalibrationResult:
    """Python implementation of the proto found in
    cirq_google.api.v2.calibration_pb2.CalibrationLayerResult for use
    in Engine calls.

    Note that, if these fields are not filled out by the calibration API,
    they will be set to the default values in the proto, as defined here:
    https://developers.google.com/protocol-buffers/docs/proto3#default
    These defaults will converted to `None` by the API client.
    """
    code: 'calibration_pb2.CalibrationLayerCode'
    error_message: Optional[str]
    token: Optional[str]
    valid_until: Optional[datetime.datetime]
    metrics: 'cirq_google.Calibration'

    @classmethod
    def _from_json_dict_(cls, code: 'calibration_pb2.CalibrationLayerCode', error_message: Optional[str], token: Optional[str], utc_valid_until: float, metrics: 'cirq_google.Calibration', **kwargs) -> 'CalibrationResult':
        """Magic method for the JSON serialization protocol."""
        valid_until = datetime.datetime.utcfromtimestamp(utc_valid_until) if utc_valid_until is not None else None
        return cls(code, error_message, token, valid_until, metrics)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        utc_valid_until = self.valid_until.replace(tzinfo=datetime.timezone.utc).timestamp() if self.valid_until is not None else None
        return {'code': self.code, 'error_message': self.error_message, 'token': self.token, 'utc_valid_until': utc_valid_until, 'metrics': self.metrics}