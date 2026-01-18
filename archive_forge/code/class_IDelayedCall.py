from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IDelayedCall(Interface):
    """
    A scheduled call.

    There are probably other useful methods we can add to this interface;
    suggestions are welcome.
    """

    def getTime() -> float:
        """
        Get time when delayed call will happen.

        @return: time in seconds since epoch (a float).
        """

    def cancel() -> None:
        """
        Cancel the scheduled call.

        @raises twisted.internet.error.AlreadyCalled: if the call has already
            happened.
        @raises twisted.internet.error.AlreadyCancelled: if the call has already
            been cancelled.
        """

    def delay(secondsLater: float) -> None:
        """
        Delay the scheduled call.

        @param secondsLater: how many seconds from its current firing time to delay

        @raises twisted.internet.error.AlreadyCalled: if the call has already
            happened.
        @raises twisted.internet.error.AlreadyCancelled: if the call has already
            been cancelled.
        """

    def reset(secondsFromNow: float) -> None:
        """
        Reset the scheduled call's timer.

        @param secondsFromNow: how many seconds from now it should fire,
            equivalent to C{.cancel()} and then doing another
            C{reactor.callLater(secondsLater, ...)}

        @raises twisted.internet.error.AlreadyCalled: if the call has already
            happened.
        @raises twisted.internet.error.AlreadyCancelled: if the call has already
            been cancelled.
        """

    def active() -> bool:
        """
        @return: True if this call is still active, False if it has been
                 called or cancelled.
        """