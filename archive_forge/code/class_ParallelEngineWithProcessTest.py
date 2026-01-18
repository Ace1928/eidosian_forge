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
class ParallelEngineWithProcessTest(EngineTaskTest, EngineMultipleResultsTest, EngineLinearFlowTest, EngineParallelFlowTest, EngineLinearAndUnorderedExceptionsTest, EngineOptionalRequirementsTest, EngineGraphFlowTest, EngineResetTests, EngineMissingDepsTest, EngineGraphConditionalFlowTest, EngineDeciderDepthTest, EngineTaskNotificationsTest, test.TestCase):
    _EXECUTOR_WORKERS = 2

    def test_correct_load(self):
        engine = self._make_engine(utils.TaskNoRequiresNoReturns)
        self.assertIsInstance(engine, eng.ParallelActionEngine)

    def _make_engine(self, flow, flow_detail=None, executor=None, store=None, **kwargs):
        if executor is None:
            executor = 'processes'
        return taskflow.engines.load(flow, flow_detail=flow_detail, backend=self.backend, engine='parallel', executor=executor, store=store, max_workers=self._EXECUTOR_WORKERS, **kwargs)

    def test_update_progress_notifications_proxied(self):
        captured = collections.defaultdict(list)

        def notify_me(event_type, details):
            captured[event_type].append(details)
        a = utils.MultiProgressingTask('a')
        a.notifier.register(a.notifier.ANY, notify_me)
        progress_chunks = list((x / 10.0 for x in range(1, 10)))
        e = self._make_engine(a, store={'progress_chunks': progress_chunks})
        e.run()
        self.assertEqual(11, len(captured[task.EVENT_UPDATE_PROGRESS]))

    def test_custom_notifications_proxied(self):
        captured = collections.defaultdict(list)

        def notify_me(event_type, details):
            captured[event_type].append(details)
        a = utils.EmittingTask('a')
        a.notifier.register(a.notifier.ANY, notify_me)
        e = self._make_engine(a)
        e.run()
        self.assertEqual(1, len(captured['hi']))
        self.assertEqual(2, len(captured[task.EVENT_UPDATE_PROGRESS]))

    def test_just_custom_notifications_proxied(self):
        captured = collections.defaultdict(list)

        def notify_me(event_type, details):
            captured[event_type].append(details)
        a = utils.EmittingTask('a')
        a.notifier.register('hi', notify_me)
        e = self._make_engine(a)
        e.run()
        self.assertEqual(1, len(captured['hi']))
        self.assertEqual(0, len(captured[task.EVENT_UPDATE_PROGRESS]))