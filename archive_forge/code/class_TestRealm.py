import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class TestRealm:
    """
    A realm which repeatedly gives out a single instance of L{MyPerspective}
    for non-anonymous logins and which gives out a new instance of L{Echoer}
    for each anonymous login.

    @ivar lastPerspective: The L{MyPerspective} most recently created and
        returned from C{requestAvatar}.

    @ivar perspectiveFactory: A one-argument callable which will be used to
        create avatars to be returned from C{requestAvatar}.
    """
    perspectiveFactory = MyPerspective
    lastPerspective = None

    def requestAvatar(self, avatarId, mind, interface):
        """
        Verify that the mind and interface supplied have the expected values
        (this should really be done somewhere else, like inside a test method)
        and return an avatar appropriate for the given identifier.
        """
        assert interface == pb.IPerspective
        assert mind == 'BRAINS!'
        if avatarId is checkers.ANONYMOUS:
            return (pb.IPerspective, Echoer(), lambda: None)
        else:
            self.lastPerspective = self.perspectiveFactory(avatarId)
            self.lastPerspective.loggedIn = True
            return (pb.IPerspective, self.lastPerspective, self.lastPerspective.logout)