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
class ParallelEngineWithThreadsTest(EngineTaskTest, EngineMultipleResultsTest, EngineLinearFlowTest, EngineParallelFlowTest, EngineLinearAndUnorderedExceptionsTest, EngineOptionalRequirementsTest, EngineGraphFlowTest, EngineResetTests, EngineMissingDepsTest, EngineGraphConditionalFlowTest, EngineCheckingTaskTest, EngineDeciderDepthTest, EngineTaskNotificationsTest, test.TestCase):
    _EXECUTOR_WORKERS = 2

    def _make_engine(self, flow, flow_detail=None, executor=None, store=None, **kwargs):
        if executor is None:
            executor = 'threads'
        return taskflow.engines.load(flow, flow_detail=flow_detail, backend=self.backend, executor=executor, engine='parallel', store=store, max_workers=self._EXECUTOR_WORKERS, **kwargs)

    def test_correct_load(self):
        engine = self._make_engine(utils.TaskNoRequiresNoReturns)
        self.assertIsInstance(engine, eng.ParallelActionEngine)

    def test_using_common_executor(self):
        flow = utils.TaskNoRequiresNoReturns(name='task1')
        executor = futurist.ThreadPoolExecutor(self._EXECUTOR_WORKERS)
        try:
            e1 = self._make_engine(flow, executor=executor)
            e2 = self._make_engine(flow, executor=executor)
            self.assertIs(e1.options['executor'], e2.options['executor'])
        finally:
            executor.shutdown(wait=True)