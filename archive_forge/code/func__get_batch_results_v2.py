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
def _get_batch_results_v2(self, results: v2.batch_pb2.BatchResult) -> Sequence[Sequence[EngineResult]]:
    return [self._get_job_results_v2(result) for result in results.results]