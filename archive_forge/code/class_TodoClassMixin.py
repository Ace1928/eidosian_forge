from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class TodoClassMixin:
    todo = 'class'

    def test_todo1(self):
        pass
    test_todo1.todo = 'method'

    def test_todo2(self):
        pass

    def test_todo3(self):
        self.fail('Deliberate Failure')
    test_todo3.todo = 'method'

    def test_todo4(self):
        self.fail('Deliberate Failure')