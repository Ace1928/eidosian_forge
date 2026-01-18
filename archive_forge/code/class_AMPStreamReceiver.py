from random import Random
from typing import Awaitable, Dict, List, TypeVar, Union
from hamcrest import (
from hypothesis import given
from hypothesis.strategies import binary, integers, just, lists, randoms, text
from twisted.internet.defer import Deferred, fail
from twisted.internet.interfaces import IProtocol
from twisted.internet.protocol import Protocol
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, connect
from twisted.trial.unittest import SynchronousTestCase
from ..stream import StreamOpen, StreamReceiver, StreamWrite, chunk, stream
from .matchers import HasSum, IsSequenceOf
class AMPStreamReceiver(AMP):
    """
    A simple AMP interface to L{StreamReceiver}.
    """

    def __init__(self, streams: StreamReceiver) -> None:
        self.streams = streams

    @StreamOpen.responder
    def streamOpen(self) -> Dict[str, object]:
        return {'streamId': self.streams.open()}

    @StreamWrite.responder
    def streamWrite(self, streamId: int, data: bytes) -> Dict[str, object]:
        self.streams.write(streamId, data)
        return {}