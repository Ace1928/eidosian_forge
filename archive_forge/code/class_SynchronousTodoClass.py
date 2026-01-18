from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousTodoClass(TodoClassMixin, SynchronousTestCase):
    pass