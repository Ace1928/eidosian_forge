import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
class ProgressTaskWithDetails(task.Task):

    def execute(self):
        details = {'progress': 0.5, 'test': 'test data', 'foo': 'bar'}
        self.notifier.notify(task.EVENT_UPDATE_PROGRESS, details)