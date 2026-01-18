from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
class XmlStreamFactoryMixinTests(GenericXmlStreamFactoryTestsMixin):
    """
    Tests for L{xmlstream.XmlStreamFactoryMixin}.
    """

    def setUp(self):
        self.factory = xmlstream.XmlStreamFactoryMixin(None, test=None)
        self.factory.protocol = DummyProtocol

    def test_buildProtocolFactoryArguments(self):
        """
        Arguments passed to the factory are passed to protocol on
        instantiation.
        """
        xs = self.factory.buildProtocol(None)
        self.assertEqual((None,), xs.args)
        self.assertEqual({'test': None}, xs.kwargs)