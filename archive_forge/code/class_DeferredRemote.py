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
class DeferredRemote(pb.Referenceable):

    def __init__(self):
        self.run = 0

    def runMe(self, arg):
        self.run = arg
        return arg + 1

    def dontRunMe(self, arg):
        assert 0, "shouldn't have been run!"

    def remote_doItLater(self):
        """
        Return a L{Deferred} to be fired on client side. When fired,
        C{self.runMe} is called.
        """
        d = Deferred()
        d.addCallbacks(self.runMe, self.dontRunMe)
        self.d = d
        return d