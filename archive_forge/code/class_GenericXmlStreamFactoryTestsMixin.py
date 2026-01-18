from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
class GenericXmlStreamFactoryTestsMixin(BootstrapMixinTests):
    """
    Generic tests for L{XmlStream} factories.
    """

    def setUp(self):
        self.factory = xmlstream.XmlStreamFactory()

    def test_buildProtocolInstallsBootstraps(self):
        """
        The protocol factory installs bootstrap event handlers on the protocol.
        """
        called = []

        def cb(data):
            called.append(data)
        self.factory.addBootstrap('//event/myevent', cb)
        xs = self.factory.buildProtocol(None)
        xs.dispatch(None, '//event/myevent')
        self.assertEqual(1, len(called))

    def test_buildProtocolStoresFactory(self):
        """
        The protocol factory is saved in the protocol.
        """
        xs = self.factory.buildProtocol(None)
        self.assertIdentical(self.factory, xs.factory)