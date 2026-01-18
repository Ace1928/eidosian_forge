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
class NewStyleCopy2(pb.Copyable, pb.RemoteCopy):
    allocated = 0
    initialized = 0
    value = 1

    def __new__(self):
        NewStyleCopy2.allocated += 1
        inst = object.__new__(self)
        inst.value = 2
        return inst

    def __init__(self):
        NewStyleCopy2.initialized += 1