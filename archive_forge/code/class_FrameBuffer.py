import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
class FrameBuffer:
    """
    A test object that converts data received from Twisted's HTTP/2 stack and
    turns it into a sequence of hyperframe frame objects.

    This is primarily used to make it easier to write and debug tests: rather
    than have to serialize the expected frames and then do byte-level
    comparison (which can be unclear in debugging output), this object makes it
    possible to work with the frames directly.

    It also ensures that headers are properly decompressed.
    """

    def __init__(self):
        self.decoder = Decoder()
        self._data = b''

    def receiveData(self, data):
        self._data += data

    def __iter__(self):
        return self

    def next(self):
        if len(self._data) < 9:
            raise StopIteration()
        frame, length = hyperframe.frame.Frame.parse_frame_header(self._data[:9])
        if len(self._data) < length + 9:
            raise StopIteration()
        frame.parse_body(memoryview(self._data[9:9 + length]))
        self._data = self._data[9 + length:]
        if isinstance(frame, hyperframe.frame.HeadersFrame):
            frame.data = self.decoder.decode(frame.data, raw=True)
        return frame
    __next__ = next