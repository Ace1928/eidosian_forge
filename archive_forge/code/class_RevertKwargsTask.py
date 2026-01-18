from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class RevertKwargsTask(task.Task):

    def execute(self, execute_arg1, execute_arg2):
        pass

    def revert(self, execute_arg1, *args, **kwargs):
        pass