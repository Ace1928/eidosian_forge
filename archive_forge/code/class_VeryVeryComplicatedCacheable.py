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
class VeryVeryComplicatedCacheable(pb.Cacheable):

    def __init__(self):
        self.x = 1
        self.y = 2
        self.foo = 3

    def setFoo4(self):
        self.foo = 4
        self.observer.callRemote('foo', 4)

    def getStateToCacheAndObserveFor(self, perspective, observer):
        self.observer = observer
        return {'x': self.x, 'y': self.y, 'foo': self.foo}

    def stoppedObserving(self, perspective, observer):
        log.msg('stopped observing')
        observer.callRemote('end')
        if observer == self.observer:
            self.observer = None