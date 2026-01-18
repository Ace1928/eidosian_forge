from typing import Optional, Sequence, Tuple
import datetime
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.engine_result import EngineResult
class NothingJob(AbstractLocalJob):
    """Blank version of AbstractLocalJob for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = quantum.ExecutionStatus.State.READY

    def execution_status(self) -> quantum.ExecutionStatus.State:
        return self._status

    def failure(self) -> Optional[Tuple[str, str]]:
        return ('failed', 'failure code')

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def batched_results_async(self) -> Sequence[Sequence[EngineResult]]:
        return []

    async def results_async(self) -> Sequence[EngineResult]:
        return []

    async def calibration_results_async(self) -> Sequence[CalibrationResult]:
        return []