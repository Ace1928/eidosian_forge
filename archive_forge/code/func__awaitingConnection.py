from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
@_machine.output()
def _awaitingConnection(self, failAfterFailures=None):
    """
        Return a deferred that will fire with the next connected protocol.

        @return: L{Deferred} that will fire with the next connected protocol.
        """
    result = Deferred()
    self._awaitingConnected.append((result, failAfterFailures))
    return result