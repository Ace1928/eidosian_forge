import datetime
from typing import Optional, Mapping, TYPE_CHECKING, Any, Dict
import numpy as np
from cirq import study
class EngineResult(study.ResultDict):
    """A ResultDict with additional job metadata.

    Please see the documentation for `cirq.ResultDict` for more information.

    Additional Attributes:
        job_id: A string job identifier.
        job_finished_time: A timestamp for when the job finished.
    """

    def __init__(self, *, job_id: str, job_finished_time: datetime.datetime, params: Optional[study.ParamResolver]=None, measurements: Optional[Mapping[str, np.ndarray]]=None, records: Optional[Mapping[str, np.ndarray]]=None):
        """Initialize the result.

        Args:
            job_id: A string job identifier.
            job_finished_time: A timestamp for when the job finished; will be converted to UTC.
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
            records: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
        """
        super().__init__(params=params, measurements=measurements, records=records)
        self.job_id = job_id
        self.job_finished_time = job_finished_time

    @classmethod
    def from_result(cls, result: 'cirq.Result', *, job_id: str, job_finished_time: datetime.datetime):
        if isinstance(result, study.ResultDict):
            return cls(params=result._params, measurements=result._measurements, records=result._records, job_id=job_id, job_finished_time=job_finished_time)
        else:
            return cls(params=result.params, measurements=result.measurements, records=result.records, job_id=job_id, job_finished_time=job_finished_time)

    def __eq__(self, other):
        if not isinstance(other, EngineResult):
            return False
        return super().__eq__(other) and self.job_id == other.job_id and (self.job_finished_time == other.job_finished_time)

    def __repr__(self) -> str:
        return f'cirq_google.EngineResult(params={self.params!r}, records={self._record_dict_repr()}, job_id={self.job_id!r}, job_finished_time={self.job_finished_time!r})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        d = super()._json_dict_()
        d['job_id'] = self.job_id
        d['job_finished_time'] = self.job_finished_time
        return d

    @classmethod
    def _from_json_dict_(cls, params, records, job_id, job_finished_time, **kwargs):
        return cls._from_packed_records(params=params, records=records, job_id=job_id, job_finished_time=job_finished_time)