from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def _getConnection(self):
    """
        Wrapper around the appropriate connect method of the reactor.

        @return: the port object returned by the connect method.
        @rtype: an object providing L{twisted.internet.interfaces.IConnector}.
        """
    return getattr(_maybeGlobalReactor(self.reactor), f'connect{self.method}')(*self.args, **self.kwargs)