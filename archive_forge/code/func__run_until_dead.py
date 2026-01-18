import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _run_until_dead(self, executor, max_dispatches=None):
    total_dispatched = 0
    if max_dispatches is None:
        max_dispatches = -1
    dispatch_gen = iter_utils.iter_forever(max_dispatches)
    is_stopped = self._wait_timeout.is_stopped
    try:
        if max_dispatches == 0:
            raise StopIteration
        fresh_period = timeutils.StopWatch(duration=self.REFRESH_PERIODICITY)
        fresh_period.start()
        while not is_stopped():
            any_dispatched = False
            if fresh_period.expired():
                ensure_fresh = True
                fresh_period.restart()
            else:
                ensure_fresh = False
            job_it = itertools.takewhile(self._can_claim_more_jobs, self._jobboard.iterjobs(ensure_fresh=ensure_fresh))
            for job in job_it:
                self._log.debug('Trying to claim job: %s', job)
                try:
                    self._jobboard.claim(job, self._name)
                except (excp.UnclaimableJob, excp.NotFound):
                    self._log.debug('Job already claimed or consumed: %s', job)
                else:
                    try:
                        fut = executor.submit(self._dispatch_job, job)
                    except RuntimeError:
                        with excutils.save_and_reraise_exception():
                            self._log.warn('Job dispatch submitting failed: %s', job)
                            self._try_finish_job(job, False)
                    else:
                        fut.job = job
                        self._dispatched.add(fut)
                        any_dispatched = True
                        fut.add_done_callback(functools.partial(self._on_job_done, job))
                        total_dispatched = next(dispatch_gen)
            if not any_dispatched and (not is_stopped()):
                self._wait_timeout.wait()
    except StopIteration:
        with excutils.save_and_reraise_exception():
            if max_dispatches >= 0 and total_dispatched >= max_dispatches:
                self._log.info('Maximum dispatch limit of %s reached', max_dispatches)