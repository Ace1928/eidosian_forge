from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousTodo(TodoMixin, SynchronousTestCase):
    pass