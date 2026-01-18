from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
class TestConsumer:
    """
    A very basic test consumer for use with the NonStreamingProducerTests.
    """

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.producer: object = None
        self.producerStreaming: bool | None = None

    def registerProducer(self, producer: object, streaming: bool) -> None:
        """
        Registers a single producer with this consumer. Just keeps track of it.

        @param producer: The producer to register.
        @param streaming: Whether the producer is a streaming one or not.
        """
        self.producer = producer
        self.producerStreaming = streaming

    def unregisterProducer(self) -> None:
        """
        Forget the producer we had previously registered.
        """
        self.producer = None
        self.producerStreaming = None

    def write(self, data: bytes) -> None:
        """
        Some data was written to the consumer: stores it for later use.

        @param data: The data to write.
        """
        self.writes.append(data)