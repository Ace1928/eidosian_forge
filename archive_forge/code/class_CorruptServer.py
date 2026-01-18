import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class CorruptServer(agent.SSHAgentServer):
    """
        A misbehaving server that returns bogus response op codes so that we can
        verify that our callbacks that deal with these op codes handle such
        miscreants.
        """

    def agentc_REQUEST_IDENTITIES(self, data):
        self.sendResponse(254, b'')

    def agentc_SIGN_REQUEST(self, data):
        self.sendResponse(254, b'')