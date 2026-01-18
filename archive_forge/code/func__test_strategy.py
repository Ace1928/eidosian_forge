import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def _test_strategy(self, schedule_strategy, nows, last_now, expected_next):
    nows = list(nows)
    ev = self.event_cls()

    def now_func():
        if len(nows) == 1:
            ev.set()
            return last_now
        return nows.pop()

    @periodics.periodic(2, run_immediately=False)
    def slow_periodic():
        pass
    callables = [(slow_periodic, None, None)]
    worker_kwargs = self.worker_kwargs.copy()
    worker_kwargs['schedule_strategy'] = schedule_strategy
    worker_kwargs['now_func'] = now_func
    w = periodics.PeriodicWorker(callables, **worker_kwargs)
    with self.create_destroy(w.start):
        ev.wait()
        w.stop()
    schedule_order = w._schedule._ordering
    self.assertEqual([(expected_next, 0)], schedule_order)