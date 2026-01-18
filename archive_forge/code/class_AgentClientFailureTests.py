import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class AgentClientFailureTests(AgentTestBase):

    def test_agentFailure(self):
        """
        verify that the client raises ConchError on AGENT_FAILURE
        """
        d = self.client.sendRequest(254, b'')
        self.pump.flush()
        return self.assertFailure(d, ConchError)