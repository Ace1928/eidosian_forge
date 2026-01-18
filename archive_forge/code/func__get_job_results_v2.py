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
def _get_job_results_v2(self, result: v2.result_pb2.Result) -> Sequence[EngineResult]:
    sweep_results = v2.results_from_proto(result)
    job_id = self.id()
    job_finished = self.update_time()
    return [EngineResult.from_result(result, job_id=job_id, job_finished_time=job_finished) for sweep_result in sweep_results for result in sweep_result]