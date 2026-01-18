import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class ServerProtocolContractWithFactoryTests(AgentTestBase):
    """
    The server protocol is stateful and so uses its factory to track state
    across requests.  This test asserts that the protocol raises if its factory
    doesn't provide the necessary storage for that state.
    """

    def test_factorySuppliesKeyStorageForServerProtocol(self):
        msg = struct.pack('!LB', 1, agent.AGENTC_REQUEST_IDENTITIES)
        del self.server.factory.__dict__['keys']
        self.assertRaises(MissingKeyStoreError, self.server.dataReceived, msg)