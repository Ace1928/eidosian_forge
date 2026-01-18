import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class AgentTestBase(unittest.TestCase):
    """
    Tests for SSHAgentServer/Client.
    """
    if agent is None or keys is None:
        skip = 'Cannot run without cryptography'

    def setUp(self):
        self.client, self.server, self.pump = iosim.connectedServerAndClient(agent.SSHAgentServer, agent.SSHAgentClient)
        self.server.factory = StubFactory()
        self.rsaPrivate = keys.Key.fromString(keydata.privateRSA_openssh)
        self.dsaPrivate = keys.Key.fromString(keydata.privateDSA_openssh)
        self.rsaPublic = keys.Key.fromString(keydata.publicRSA_openssh)
        self.dsaPublic = keys.Key.fromString(keydata.publicDSA_openssh)