import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class UnimplementedVersionOneServerTests(AgentTestBase):
    """
    Tests for methods with no-op implementations on the server. We need these
    for clients, such as openssh, that try v1 methods before going to v2.

    Because the client doesn't expose these operations with nice method names,
    we invoke sendRequest directly with an op code.
    """

    def test_agentc_REQUEST_RSA_IDENTITIES(self):
        """
        assert that we get the correct op code for an RSA identities request
        """
        d = self.client.sendRequest(agent.AGENTC_REQUEST_RSA_IDENTITIES, b'')
        self.pump.flush()

        def _cb(packet):
            self.assertEqual(agent.AGENT_RSA_IDENTITIES_ANSWER, ord(packet[0:1]))
        return d.addCallback(_cb)

    def test_agentc_REMOVE_RSA_IDENTITY(self):
        """
        assert that we get the correct op code for an RSA remove identity request
        """
        d = self.client.sendRequest(agent.AGENTC_REMOVE_RSA_IDENTITY, b'')
        self.pump.flush()
        return d.addCallback(self.assertEqual, b'')

    def test_agentc_REMOVE_ALL_RSA_IDENTITIES(self):
        """
        assert that we get the correct op code for an RSA remove all identities
        request.
        """
        d = self.client.sendRequest(agent.AGENTC_REMOVE_ALL_RSA_IDENTITIES, b'')
        self.pump.flush()
        return d.addCallback(self.assertEqual, b'')