import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class GMPYInstallDeprecationTests(unittest.TestCase):
    """
    Tests for the deprecation of former GMPY accidental public API.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def test_deprecated(self):
        """
        L{twisted.conch.ssh.common.install} is deprecated.
        """
        common.install()
        warnings = self.flushWarnings([self.test_deprecated])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'twisted.conch.ssh.common.install was deprecated in Twisted 16.5.0')