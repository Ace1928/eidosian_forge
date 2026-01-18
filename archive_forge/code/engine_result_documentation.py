import datetime
from typing import Optional, Mapping, TYPE_CHECKING, Any, Dict
import numpy as np
from cirq import study
Initialize the result.

        Args:
            job_id: A string job identifier.
            job_finished_time: A timestamp for when the job finished; will be converted to UTC.
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
            records: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
        