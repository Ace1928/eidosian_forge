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
def _get_job_results_v1(self, result: v1.program_pb2.Result) -> Sequence[EngineResult]:
    job_id = self.id()
    job_finished = self.update_time()
    trial_results = []
    for sweep_result in result.sweep_results:
        sweep_repetitions = sweep_result.repetitions
        key_sizes = [(m.key, len(m.qubits)) for m in sweep_result.measurement_keys]
        for result in sweep_result.parameterized_results:
            data = result.measurement_results
            measurements = v1.unpack_results(data, sweep_repetitions, key_sizes)
            trial_results.append(EngineResult(params=cirq.ParamResolver(result.params.assignments), measurements=measurements, job_id=job_id, job_finished_time=job_finished))
    return trial_results