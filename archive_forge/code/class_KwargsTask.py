from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class KwargsTask(task.Task):

    def execute(self, spam, **kwargs):
        pass