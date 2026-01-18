import concurrent.futures
import datetime
from typing import cast, List, Optional, Sequence, Tuple
import duet
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.engine_result import EngineResult
def _to_engine_results(batch_results: Sequence[Sequence['cirq.Result']], *, job_id: str, job_finished_time: Optional[datetime.datetime]=None) -> List[List[EngineResult]]:
    """Convert cirq.Result from simulators into (simulated) EngineResults."""
    if job_finished_time is None:
        job_finished_time = datetime.datetime.now(tz=datetime.timezone.utc)
    return [[EngineResult.from_result(result, job_id=job_id, job_finished_time=job_finished_time) for result in batch] for batch in batch_results]