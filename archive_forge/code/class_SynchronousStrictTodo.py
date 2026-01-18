from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousStrictTodo(StrictTodoMixin, SynchronousTestCase):
    pass