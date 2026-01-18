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
class ChunkTests(SynchronousTestCase):
    """
    Tests for ``chunk``.
    """

    @given(data=text(), chunkSize=integers(min_value=1))
    def test_chunk(self, data, chunkSize):
        """
        L{chunk} returns an iterable of L{str} where each element is no
        longer than the given limit.  The concatenation of the strings is also
        equal to the original input string.
        """
        chunks = list(chunk(data, chunkSize))
        assert_that(chunks, all_of(IsSequenceOf(has_length(less_than_or_equal_to(chunkSize))), HasSum(equal_to(data), data[:0])))