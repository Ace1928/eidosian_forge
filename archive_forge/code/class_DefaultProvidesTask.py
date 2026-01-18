from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class DefaultProvidesTask(task.Task):
    default_provides = 'def'

    def execute(self):
        return None