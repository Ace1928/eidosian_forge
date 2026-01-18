import logging
import os
from taskflow import exceptions
from taskflow.listeners import base
from taskflow import states
def _has_been_lost(self):
    try:
        job_state = self._job.state
        job_owner = self._board.find_owner(self._job)
    except (exceptions.NotFound, exceptions.JobFailure):
        return True
    else:
        if job_state == states.UNCLAIMED or self._owner != job_owner:
            return True
        else:
            return False