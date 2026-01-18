from zope.interface import implementer
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import (
from twisted.cred.credentials import (
from twisted.cred.error import LoginDenied
from twisted.cred.portal import Portal
from twisted.internet.interfaces import IReactorProcess
from twisted.python.fakepwd import UserDatabase
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from .test_session import StubClient, StubConnection
@implementer(IReactorProcess)
class MockProcessSpawner:
    """
    An L{IReactorProcess} that logs calls to C{spawnProcess}.
    """

    def __init__(self):
        self._spawnProcessCalls = []

    def spawnProcess(self, processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        """
        Log a call to C{spawnProcess}. Do not actually spawn a process.
        """
        self._spawnProcessCalls.append({'processProtocol': processProtocol, 'executable': executable, 'args': args, 'env': env, 'path': path, 'uid': uid, 'gid': gid, 'usePTY': usePTY, 'childFDs': childFDs})