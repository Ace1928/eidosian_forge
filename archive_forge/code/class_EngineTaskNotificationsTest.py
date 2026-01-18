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
class EngineTaskNotificationsTest(object):

    def test_run_capture_task_notifications(self):
        captured = collections.defaultdict(list)

        def do_capture(bound_name, event_type, details):
            progress_capture = captured[bound_name]
            progress_capture.append(details)
        flow = lf.Flow('flow')
        work_1 = utils.MultiProgressingTask('work-1')
        work_1.notifier.register(task.EVENT_UPDATE_PROGRESS, functools.partial(do_capture, 'work-1'))
        work_2 = utils.MultiProgressingTask('work-2')
        work_2.notifier.register(task.EVENT_UPDATE_PROGRESS, functools.partial(do_capture, 'work-2'))
        flow.add(work_1, work_2)
        progress_chunks = tuple([0.2, 0.5, 0.8])
        engine = self._make_engine(flow, store={'progress_chunks': progress_chunks})
        engine.run()
        expected = [{'progress': 0.0}, {'progress': 0.2}, {'progress': 0.5}, {'progress': 0.8}, {'progress': 1.0}]
        for name in ['work-1', 'work-2']:
            self.assertEqual(expected, captured[name])