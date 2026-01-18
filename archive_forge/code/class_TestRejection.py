import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
class TestRejection(testscenarios.TestWithScenarios, base.TestCase):
    rejector = rejection.reject_when_reached(1)
    scenarios = [('green', {'executor_cls': futurist.GreenThreadPoolExecutor, 'executor_kwargs': {'check_and_reject': rejector, 'max_workers': 1}, 'event_cls': green_threading.Event}), ('thread', {'executor_cls': futurist.ThreadPoolExecutor, 'executor_kwargs': {'check_and_reject': rejector, 'max_workers': 1}, 'event_cls': threading.Event})]

    def setUp(self):
        super(TestRejection, self).setUp()
        self.executor = self.executor_cls(**self.executor_kwargs)
        self.addCleanup(self.executor.shutdown, wait=True)

    def test_rejection(self):
        ev = self.event_cls()
        ev_thread_started = self.event_cls()
        self.addCleanup(ev.set)

        def wait_until_set(check_delay):
            ev_thread_started.set()
            while not ev.is_set():
                ev.wait(check_delay)
        self.executor.submit(wait_until_set, 0.1)
        ev_thread_started.wait()
        self.executor.submit(wait_until_set, 0.1)
        self.assertRaises(futurist.RejectedSubmission, self.executor.submit, returns_one)