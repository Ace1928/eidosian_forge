import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
class RetryParallelExecutionTest(utils.EngineTestBase):

    def test_when_subflow_fails_revert_running_tasks(self):
        waiting_task = utils.WaitForOneFromTask('task1', 'task2', [st.SUCCESS, st.FAILURE])
        flow = uf.Flow('flow-1', retry.Times(3, 'r', provides='x')).add(waiting_task, utils.ConditionalTask('task2'))
        engine = self._make_engine(flow)
        engine.atom_notifier.register('*', waiting_task.callback)
        engine.storage.inject({'y': 2})
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        self.assertEqual({'y': 2, 'x': 2}, engine.storage.fetch_all())
        expected = ['r.r RUNNING', 'r.r SUCCESS(1)', 'task1.t RUNNING', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot!)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'task1.t SUCCESS(5)', 'task1.t REVERTING', 'task1.t REVERTED(None)', 'r.r RETRYING', 'task1.t PENDING', 'task2.t PENDING', 'r.r RUNNING', 'r.r SUCCESS(2)', 'task1.t RUNNING', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'task1.t SUCCESS(5)']
        self.assertCountEqual(capturer.values, expected)

    def test_when_subflow_fails_revert_success_tasks(self):
        waiting_task = utils.WaitForOneFromTask('task2', 'task1', [st.SUCCESS, st.FAILURE])
        flow = uf.Flow('flow-1', retry.Times(3, 'r', provides='x')).add(utils.ProgressingTask('task1'), lf.Flow('flow-2').add(waiting_task, utils.ConditionalTask('task3')))
        engine = self._make_engine(flow)
        engine.atom_notifier.register('*', waiting_task.callback)
        engine.storage.inject({'y': 2})
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        self.assertEqual({'y': 2, 'x': 2}, engine.storage.fetch_all())
        expected = ['r.r RUNNING', 'r.r SUCCESS(1)', 'task1.t RUNNING', 'task2.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t SUCCESS(5)', 'task3.t RUNNING', 'task3.t FAILURE(Failure: RuntimeError: Woot!)', 'task3.t REVERTING', 'task1.t REVERTING', 'task3.t REVERTED(None)', 'task1.t REVERTED(None)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r.r RETRYING', 'task1.t PENDING', 'task2.t PENDING', 'task3.t PENDING', 'r.r RUNNING', 'r.r SUCCESS(2)', 'task1.t RUNNING', 'task2.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t SUCCESS(5)', 'task3.t RUNNING', 'task3.t SUCCESS(None)']
        self.assertCountEqual(capturer.values, expected)