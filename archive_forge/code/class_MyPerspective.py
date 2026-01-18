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
@implementer(pb.IPerspective)
class MyPerspective(pb.Avatar):
    """
    @ivar loggedIn: set to C{True} when the avatar is logged in.
    @type loggedIn: C{bool}

    @ivar loggedOut: set to C{True} when the avatar is logged out.
    @type loggedOut: C{bool}
    """
    loggedIn = loggedOut = False

    def __init__(self, avatarId):
        self.avatarId = avatarId

    def perspective_getAvatarId(self):
        """
        Return the avatar identifier which was used to access this avatar.
        """
        return self.avatarId

    def perspective_getViewPoint(self):
        return MyView()

    def perspective_add(self, a, b):
        """
        Add the given objects and return the result.  This is a method
        unavailable on L{Echoer}, so it can only be invoked by authenticated
        users who received their avatar from L{TestRealm}.
        """
        return a + b

    def logout(self):
        self.loggedOut = True