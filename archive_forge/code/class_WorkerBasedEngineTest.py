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
class WorkerBasedEngineTest(EngineTaskTest, EngineMultipleResultsTest, EngineLinearFlowTest, EngineParallelFlowTest, EngineLinearAndUnorderedExceptionsTest, EngineOptionalRequirementsTest, EngineGraphFlowTest, EngineResetTests, EngineMissingDepsTest, EngineGraphConditionalFlowTest, EngineDeciderDepthTest, EngineTaskNotificationsTest, test.TestCase):

    def setUp(self):
        super(WorkerBasedEngineTest, self).setUp()
        shared_conf = {'exchange': 'test', 'transport': 'memory', 'transport_options': {'polling_interval': 0.01}}
        worker_conf = shared_conf.copy()
        worker_conf.update({'topic': 'my-topic', 'tasks': [utils.__name__]})
        self.engine_conf = shared_conf.copy()
        self.engine_conf.update({'engine': 'worker-based', 'topics': tuple([worker_conf['topic']])})
        self.worker = wkr.Worker(**worker_conf)
        self.worker_thread = tu.daemon_thread(self.worker.run)
        self.worker_thread.start()
        self.addCleanup(self.worker_thread.join)
        self.addCleanup(self.worker.stop)
        self.worker.wait()

    def _make_engine(self, flow, flow_detail=None, store=None, **kwargs):
        kwargs.update(self.engine_conf)
        return taskflow.engines.load(flow, flow_detail=flow_detail, backend=self.backend, store=store, **kwargs)

    def test_correct_load(self):
        engine = self._make_engine(utils.TaskNoRequiresNoReturns)
        self.assertIsInstance(engine, w_eng.WorkerBasedActionEngine)