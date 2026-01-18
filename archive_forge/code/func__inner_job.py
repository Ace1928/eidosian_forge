import datetime
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
from google.protobuf import any_pb2
import cirq
from cirq_google.engine import abstract_job, calibration, engine_client
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.engine.engine_result import EngineResult
from cirq_google.api import v1, v2
def _inner_job(self) -> quantum.QuantumJob:
    if self._job is None:
        self._job = self._get_job()
    return self._job