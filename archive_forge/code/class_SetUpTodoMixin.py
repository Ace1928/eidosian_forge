from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SetUpTodoMixin:

    def setUp(self):
        raise RuntimeError('deliberate error')

    def test_todo1(self):
        pass
    test_todo1.todo = 'setUp todo1'