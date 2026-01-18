import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
def _process_scheduled():
    with self._waiter:
        while not self._schedule and (not self._tombstone.is_set()) and (not self._immediates):
            self._waiter.wait(self.MAX_LOOP_IDLE)
        if self._tombstone.is_set():
            return
        if self._immediates:
            return
        submitted_at = now = self._now_func()
        next_run, index = self._schedule.pop()
        when_next = next_run - now
        if when_next <= 0:
            work = self._works[index]
            self._log.debug("Submitting periodic callback '%s'", work.name)
            try:
                fut = executor.submit(runner.run, work)
            except _SCHEDULE_RETRY_EXCEPTIONS as exc:
                delay = self._RESCHEDULE_DELAY + rnd.random() * self._RESCHEDULE_JITTER
                self._log.error("Failed to submit periodic callback '%s', retrying after %.2f sec. Error: %s", work.name, delay, exc)
                self._schedule.push(self._now_func() + delay, index)
            else:
                barrier.incr()
                fut.add_done_callback(functools.partial(_on_done, PERIODIC, work, index, submitted_at))
        else:
            self._schedule.push(next_run, index)
            when_next = min(when_next, self.MAX_LOOP_IDLE)
            self._waiter.wait(when_next)