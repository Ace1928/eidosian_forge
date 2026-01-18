import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class StubFactory:
    """
    Mock factory that provides the keys attribute required by the
    SSHAgentServerProtocol
    """

    def __init__(self):
        self.keys = {}