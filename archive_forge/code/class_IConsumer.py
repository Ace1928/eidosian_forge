from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IConsumer(Interface):
    """
    A consumer consumes data from a producer.
    """

    def registerProducer(producer: 'IProducer', streaming: bool) -> None:
        """
        Register to receive data from a producer.

        This sets self to be a consumer for a producer.  When this object runs
        out of data (as when a send(2) call on a socket succeeds in moving the
        last data from a userspace buffer into a kernelspace buffer), it will
        ask the producer to resumeProducing().

        For L{IPullProducer} providers, C{resumeProducing} will be called once
        each time data is required.

        For L{IPushProducer} providers, C{pauseProducing} will be called
        whenever the write buffer fills up and C{resumeProducing} will only be
        called when it empties.  The consumer will only call C{resumeProducing}
        to balance a previous C{pauseProducing} call; the producer is assumed
        to start in an un-paused state.

        @param streaming: C{True} if C{producer} provides L{IPushProducer},
            C{False} if C{producer} provides L{IPullProducer}.

        @raise RuntimeError: If a producer is already registered.
        """

    def unregisterProducer() -> None:
        """
        Stop consuming data from a producer, without disconnecting.
        """

    def write(data: bytes) -> None:
        """
        The producer will write data by calling this method.

        The implementation must be non-blocking and perform whatever
        buffering is necessary.  If the producer has provided enough data
        for now and it is a L{IPushProducer}, the consumer may call its
        C{pauseProducing} method.
        """