from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class MapFunctorTaskTest(test.TestCase):

    def test_invalid_functor(self):
        self.assertRaises(ValueError, task.MapFunctorTask, 2, requires=5)
        self.assertRaises(ValueError, task.MapFunctorTask, lambda: None, requires=5)
        self.assertRaises(ValueError, task.MapFunctorTask, lambda x, y: None, requires=5)

    def test_functor_invalid_requires(self):
        self.assertRaises(TypeError, task.MapFunctorTask, lambda x: None, requires=1)