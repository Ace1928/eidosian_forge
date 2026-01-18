import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class AgentIdentityRequestsTests(AgentTestBase):
    """
    Test operations against a server with identities already loaded.
    """

    def setUp(self):
        AgentTestBase.setUp(self)
        self.server.factory.keys[self.dsaPrivate.blob()] = (self.dsaPrivate, b'a comment')
        self.server.factory.keys[self.rsaPrivate.blob()] = (self.rsaPrivate, b'another comment')

    def test_signDataRSA(self):
        """
        Sign data with an RSA private key and then verify it with the public
        key.
        """
        d = self.client.signData(self.rsaPublic.blob(), b'John Hancock')
        self.pump.flush()
        signature = self.successResultOf(d)
        expected = self.rsaPrivate.sign(b'John Hancock')
        self.assertEqual(expected, signature)
        self.assertTrue(self.rsaPublic.verify(signature, b'John Hancock'))

    def test_signDataDSA(self):
        """
        Sign data with a DSA private key and then verify it with the public
        key.
        """
        d = self.client.signData(self.dsaPublic.blob(), b'John Hancock')
        self.pump.flush()

        def _check(sig):
            self.assertTrue(self.dsaPublic.verify(sig, b'John Hancock'))
        return d.addCallback(_check)

    def test_signDataRSAErrbackOnUnknownBlob(self):
        """
        Assert that we get an errback if we try to sign data using a key that
        wasn't added.
        """
        del self.server.factory.keys[self.rsaPublic.blob()]
        d = self.client.signData(self.rsaPublic.blob(), b'John Hancock')
        self.pump.flush()
        return self.assertFailure(d, ConchError)

    def test_requestIdentities(self):
        """
        Assert that we get all of the keys/comments that we add when we issue a
        request for all identities.
        """
        d = self.client.requestIdentities()
        self.pump.flush()

        def _check(keyt):
            expected = {}
            expected[self.dsaPublic.blob()] = b'a comment'
            expected[self.rsaPublic.blob()] = b'another comment'
            received = {}
            for k in keyt:
                received[keys.Key.fromString(k[0], type='blob').blob()] = k[1]
            self.assertEqual(expected, received)
        return d.addCallback(_check)