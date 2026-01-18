from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousSetUpTodo(SetUpTodoMixin, SynchronousTestCase):
    pass