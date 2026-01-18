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
class StreamTests(SynchronousTestCase):
    """
    Tests for L{stream}.
    """

    @given(lists(binary()))
    def test_stream(self, chunks: List[bytes]) -> None:
        """
        All of the chunks passed to L{stream} are sent in order over a
        stream using the given AMP connection.
        """
        sender = AMP()
        streams = StreamReceiver()
        streamId = interact(AMPStreamReceiver(streams), sender, stream(sender, iter(chunks)))
        assert_that(streams.finish(streamId), is_(equal_to(chunks)))