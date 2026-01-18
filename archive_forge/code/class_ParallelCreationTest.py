import futurist
import testtools
from taskflow.engines.action_engine import engine
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as pu
class ParallelCreationTest(test.TestCase):

    @staticmethod
    def _create_engine(**kwargs):
        flow = lf.Flow('test-flow').add(utils.DummyTask())
        backend = backends.fetch({'connection': 'memory'})
        flow_detail = pu.create_flow_detail(flow, backend=backend)
        options = kwargs.copy()
        return engine.ParallelActionEngine(flow, flow_detail, backend, options)

    def test_thread_string_creation(self):
        for s in ['threads', 'threaded', 'thread']:
            eng = self._create_engine(executor=s)
            self.assertIsInstance(eng._task_executor, executor.ParallelThreadTaskExecutor)

    def test_process_string_creation(self):
        for s in ['process', 'processes']:
            eng = self._create_engine(executor=s)
            self.assertIsInstance(eng._task_executor, process_executor.ParallelProcessTaskExecutor)

    def test_thread_executor_creation(self):
        with futurist.ThreadPoolExecutor(1) as e:
            eng = self._create_engine(executor=e)
            self.assertIsInstance(eng._task_executor, executor.ParallelThreadTaskExecutor)

    def test_process_executor_creation(self):
        with futurist.ProcessPoolExecutor(1) as e:
            eng = self._create_engine(executor=e)
            self.assertIsInstance(eng._task_executor, process_executor.ParallelProcessTaskExecutor)

    @testtools.skipIf(not eu.EVENTLET_AVAILABLE, 'eventlet is not available')
    def test_green_executor_creation(self):
        with futurist.GreenThreadPoolExecutor(1) as e:
            eng = self._create_engine(executor=e)
            self.assertIsInstance(eng._task_executor, executor.ParallelThreadTaskExecutor)

    def test_sync_executor_creation(self):
        with futurist.SynchronousExecutor() as e:
            eng = self._create_engine(executor=e)
            self.assertIsInstance(eng._task_executor, executor.ParallelThreadTaskExecutor)

    def test_invalid_creation(self):
        self.assertRaises(ValueError, self._create_engine, executor='crap')
        self.assertRaises(TypeError, self._create_engine, executor=2)
        self.assertRaises(TypeError, self._create_engine, executor=object())