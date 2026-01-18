import collections
import contextlib
import functools
import threading
import futurist
import testtools
import taskflow.engines
from taskflow.engines.action_engine import engine as eng
from taskflow.engines.worker_based import engine as w_eng
from taskflow.engines.worker_based import worker as wkr
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow.persistence import models
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.types import graph as gr
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils as tu
class EngineTaskTest(object):

    def test_run_task_as_flow(self):
        flow = utils.ProgressingTask(name='task1')
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)

    def test_run_task_with_flow_notifications(self):
        flow = utils.ProgressingTask(name='task1')
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine) as capturer:
            engine.run()
        expected = ['task1.f RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'task1.f SUCCESS']
        self.assertEqual(expected, capturer.values)

    def test_failing_task_with_flow_notifications(self):
        values = []
        flow = utils.FailingTask('fail')
        engine = self._make_engine(flow)
        expected = ['fail.f RUNNING', 'fail.t RUNNING', 'fail.t FAILURE(Failure: RuntimeError: Woot!)', 'fail.t REVERTING', 'fail.t REVERTED(None)', 'fail.f REVERTED']
        with utils.CaptureListener(engine, values=values) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        self.assertEqual(expected, capturer.values)
        self.assertEqual(states.REVERTED, engine.storage.get_flow_state())
        with utils.CaptureListener(engine, values=values) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        now_expected = list(expected)
        now_expected.extend(['fail.t PENDING', 'fail.f PENDING'])
        now_expected.extend(expected)
        self.assertEqual(now_expected, values)
        self.assertEqual(states.REVERTED, engine.storage.get_flow_state())

    def test_invalid_flow_raises(self):

        def compile_bad(value):
            engine = self._make_engine(value)
            engine.compile()
        value = 'i am string, not task/flow, sorry'
        err = self.assertRaises(TypeError, compile_bad, value)
        self.assertIn(value, str(err))

    def test_invalid_flow_raises_from_run(self):

        def run_bad(value):
            engine = self._make_engine(value)
            engine.run()
        value = 'i am string, not task/flow, sorry'
        err = self.assertRaises(TypeError, run_bad, value)
        self.assertIn(value, str(err))

    def test_nasty_failing_task_exception_reraised(self):
        flow = utils.NastyFailingTask()
        engine = self._make_engine(flow)
        self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)