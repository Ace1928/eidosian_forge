import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
class DialectTests(BananaTestBase):
    """
    Tests for Banana's handling of dialects.
    """
    vocab = b'remote'
    legalPbItem = bytes((banana.Banana.outgoingVocabulary[vocab],)) + banana.VOCAB
    illegalPbItem = bytes((122,)) + banana.VOCAB

    def test_dialectNotSet(self):
        """
        If no dialect has been selected and a PB VOCAB item is received,
        L{NotImplementedError} is raised.
        """
        self.assertRaises(NotImplementedError, self.enc.dataReceived, self.legalPbItem)

    def test_receivePb(self):
        """
        If the PB dialect has been selected, a PB VOCAB item is accepted.
        """
        selectDialect(self.enc, b'pb')
        self.enc.dataReceived(self.legalPbItem)
        self.assertEqual(self.result, self.vocab)

    def test_receiveIllegalPb(self):
        """
        If the PB dialect has been selected and an unrecognized PB VOCAB item
        is received, L{banana.Banana.dataReceived} raises L{KeyError}.
        """
        selectDialect(self.enc, b'pb')
        self.assertRaises(KeyError, self.enc.dataReceived, self.illegalPbItem)

    def test_sendPb(self):
        """
        if pb dialect is selected, the sender must be able to send things in
        that dialect.
        """
        selectDialect(self.enc, b'pb')
        self.enc.sendEncoded(self.vocab)
        self.assertEqual(self.legalPbItem, self.io.getvalue())