from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class FunctorTaskTest(test.TestCase):

    def test_creation_with_version(self):
        version = (2, 0)
        f_task = task.FunctorTask(lambda: None, version=version)
        self.assertEqual(version, f_task.version)

    def test_execute_not_callable(self):
        self.assertRaises(ValueError, task.FunctorTask, 2)

    def test_revert_not_callable(self):
        self.assertRaises(ValueError, task.FunctorTask, lambda: None, revert=2)