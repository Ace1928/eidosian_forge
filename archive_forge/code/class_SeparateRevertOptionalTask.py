from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class SeparateRevertOptionalTask(task.Task):

    def execute(self, execute_arg=None):
        pass

    def revert(self, result, flow_failures, revert_arg=None):
        pass