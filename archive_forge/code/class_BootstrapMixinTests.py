from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
class BootstrapMixinTests(unittest.TestCase):
    """
    Tests for L{xmlstream.BootstrapMixin}.

    @ivar factory: Instance of the factory or mixin under test.
    """

    def setUp(self):
        self.factory = xmlstream.BootstrapMixin()

    def test_installBootstraps(self):
        """
        Dispatching an event fires registered bootstrap observers.
        """
        called = []

        def cb(data):
            called.append(data)
        dispatcher = DummyProtocol()
        self.factory.addBootstrap('//event/myevent', cb)
        self.factory.installBootstraps(dispatcher)
        dispatcher.dispatch(None, '//event/myevent')
        self.assertEqual(1, len(called))

    def test_addAndRemoveBootstrap(self):
        """
        Test addition and removal of a bootstrap event handler.
        """
        called = []

        def cb(data):
            called.append(data)
        self.factory.addBootstrap('//event/myevent', cb)
        self.factory.removeBootstrap('//event/myevent', cb)
        dispatcher = DummyProtocol()
        self.factory.installBootstraps(dispatcher)
        dispatcher.dispatch(None, '//event/myevent')
        self.assertFalse(called)