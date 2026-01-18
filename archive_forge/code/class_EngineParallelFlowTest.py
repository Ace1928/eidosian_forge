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
class EngineParallelFlowTest(utils.EngineTestBase):

    def test_run_empty_unordered_flow(self):
        flow = uf.Flow('p-1')
        engine = self._make_engine(flow)
        self.assertEqual(_EMPTY_TRANSITIONS, list(engine.run_iter()))

    def test_parallel_flow_with_priority(self):
        flow = uf.Flow('p-1')
        for i in range(0, 10):
            t = utils.ProgressingTask(name='task%s' % i)
            t.priority = i
            flow.add(t)
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task9.t RUNNING', 'task8.t RUNNING', 'task7.t RUNNING', 'task6.t RUNNING', 'task5.t RUNNING', 'task4.t RUNNING', 'task3.t RUNNING', 'task2.t RUNNING', 'task1.t RUNNING', 'task0.t RUNNING']
        gotten = capturer.values[0:10]
        self.assertEqual(expected, gotten)

    def test_parallel_flow_one_task(self):
        flow = uf.Flow('p-1').add(utils.ProgressingTask(name='task1', provides='a'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual({'a': 5}, engine.storage.fetch_all())

    def test_parallel_flow_two_tasks(self):
        flow = uf.Flow('p-2').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = set(['task2.t SUCCESS(5)', 'task2.t RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(5)'])
        self.assertEqual(expected, set(capturer.values))

    def test_parallel_revert(self):
        flow = uf.Flow('p-r-3').add(utils.TaskNoRequiresNoReturns(name='task1'), utils.FailingTask(name='fail'), utils.TaskNoRequiresNoReturns(name='task2'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        self.assertIn('fail.t FAILURE(Failure: RuntimeError: Woot!)', capturer.values)

    def test_parallel_revert_exception_is_reraised(self):
        flow = lf.Flow('p-r-r-l').add(uf.Flow('p-r-r').add(utils.TaskNoRequiresNoReturns(name='task1'), utils.NastyTask()), utils.FailingTask())
        engine = self._make_engine(flow)
        self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)

    def test_sequential_flow_two_tasks_with_resumption(self):
        flow = lf.Flow('lf-2-r').add(utils.ProgressingTask(name='task1', provides='x1'), utils.ProgressingTask(name='task2', provides='x2'))
        lb, fd = p_utils.temporary_flow_detail(self.backend)
        td = models.TaskDetail(name='task1', uuid='42')
        td.state = states.SUCCESS
        td.results = 17
        fd.add(td)
        with contextlib.closing(self.backend.get_connection()) as conn:
            fd.update(conn.update_flow_details(fd))
            td.update(conn.update_atom_details(td))
        engine = self._make_engine(flow, fd)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task2.t RUNNING', 'task2.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual({'x1': 17, 'x2': 5}, engine.storage.fetch_all())