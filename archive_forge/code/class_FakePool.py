import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class FakePool:
    """
    A fake L{ConnectionPool} for tests.

    @ivar connectionFactory: factory for making connections returned by the
        C{connect} method.
    @type connectionFactory: any callable
    """
    reconnect = True
    noisy = True

    def __init__(self, connectionFactory):
        self.connectionFactory = connectionFactory

    def connect(self):
        """
        Return an instance of C{self.connectionFactory}.
        """
        return self.connectionFactory()

    def disconnect(self, connection):
        """
        Do nothing.
        """