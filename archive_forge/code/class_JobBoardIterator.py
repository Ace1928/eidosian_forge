import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
class JobBoardIterator(object):
    """Iterator over a jobboard that iterates over potential jobs.

    It provides the following attributes:

    * ``only_unclaimed``: boolean that indicates whether to only iterate
      over unclaimed jobs
    * ``ensure_fresh``: boolean that requests that during every fetch of a new
      set of jobs this will cause the iterator to force the backend to
      refresh (ensuring that the jobboard has the most recent job listings)
    * ``board``: the board this iterator was created from
    """
    _UNCLAIMED_JOB_STATES = (states.UNCLAIMED,)
    _JOB_STATES = (states.UNCLAIMED, states.COMPLETE, states.CLAIMED)

    def __init__(self, board, logger, board_fetch_func=None, board_removal_func=None, only_unclaimed=False, ensure_fresh=False):
        self._board = board
        self._logger = logger
        self._board_removal_func = board_removal_func
        self._board_fetch_func = board_fetch_func
        self._fetched = False
        self._jobs = collections.deque()
        self.only_unclaimed = only_unclaimed
        self.ensure_fresh = ensure_fresh

    @property
    def board(self):
        """The board this iterator was created from."""
        return self._board

    def __iter__(self):
        return self

    def _next_job(self):
        if self.only_unclaimed:
            allowed_states = self._UNCLAIMED_JOB_STATES
        else:
            allowed_states = self._JOB_STATES
        job = None
        while self._jobs and job is None:
            maybe_job = self._jobs.popleft()
            try:
                if maybe_job.state in allowed_states:
                    job = maybe_job
            except excp.JobFailure:
                self._logger.warn("Failed determining the state of job '%s'", maybe_job, exc_info=True)
            except excp.NotFound:
                if self._board_removal_func is not None:
                    self._board_removal_func(maybe_job)
        return job

    def __next__(self):
        if not self._jobs:
            if not self._fetched:
                if self._board_fetch_func is not None:
                    self._jobs.extend(self._board_fetch_func(ensure_fresh=self.ensure_fresh))
                self._fetched = True
        job = self._next_job()
        if job is None:
            raise StopIteration
        else:
            return job