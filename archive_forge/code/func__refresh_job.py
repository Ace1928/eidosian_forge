import time
import warnings
from typing import Dict, Sequence, Union, Optional, TYPE_CHECKING
from cirq_ionq import ionq_exceptions, results
from cirq._doc import document
import cirq
def _refresh_job(self):
    """If the last fetched job is not terminal, gets the job from the API."""
    if self._job['status'] not in self.TERMINAL_STATES:
        self._job = self._client.get_job(self.job_id())