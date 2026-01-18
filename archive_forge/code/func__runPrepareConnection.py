from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def _runPrepareConnection(self, protocol):
    """
        Run any C{prepareConnection} callback with the connected protocol,
        ignoring its return value but propagating any failure.

        @param protocol: The protocol of the connection.
        @type protocol: L{IProtocol}

        @return: Either:

            - A L{Deferred} that succeeds with the protocol when the
              C{prepareConnection} callback has executed successfully.

            - A L{Deferred} that fails when the C{prepareConnection} callback
              throws or returns a failed L{Deferred}.

            - The protocol, when no C{prepareConnection} callback is defined.
        """
    if self._prepareConnection:
        return maybeDeferred(self._prepareConnection, protocol).addCallback(lambda _: protocol)
    return protocol