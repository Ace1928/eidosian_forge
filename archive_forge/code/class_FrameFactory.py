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
class FrameFactory:
    """
    A class containing lots of helper methods and state to build frames. This
    allows test cases to easily build correct HTTP/2 frames to feed to
    hyper-h2.
    """

    def __init__(self):
        self.encoder = Encoder()

    def refreshEncoder(self):
        self.encoder = Encoder()

    def clientConnectionPreface(self):
        return b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'

    def buildHeadersFrame(self, headers, flags=[], streamID=1, **priorityKwargs):
        """
        Builds a single valid headers frame out of the contained headers.
        """
        f = hyperframe.frame.HeadersFrame(streamID)
        f.data = self.encoder.encode(headers)
        f.flags.add('END_HEADERS')
        for flag in flags:
            f.flags.add(flag)
        for k, v in priorityKwargs.items():
            setattr(f, k, v)
        return f

    def buildDataFrame(self, data, flags=None, streamID=1):
        """
        Builds a single data frame out of a chunk of data.
        """
        flags = set(flags) if flags is not None else set()
        f = hyperframe.frame.DataFrame(streamID)
        f.data = data
        f.flags = flags
        return f

    def buildSettingsFrame(self, settings, ack=False):
        """
        Builds a single settings frame.
        """
        f = hyperframe.frame.SettingsFrame(0)
        if ack:
            f.flags.add('ACK')
        f.settings = settings
        return f

    def buildWindowUpdateFrame(self, streamID, increment):
        """
        Builds a single WindowUpdate frame.
        """
        f = hyperframe.frame.WindowUpdateFrame(streamID)
        f.window_increment = increment
        return f

    def buildGoAwayFrame(self, lastStreamID, errorCode=0, additionalData=b''):
        """
        Builds a single GOAWAY frame.
        """
        f = hyperframe.frame.GoAwayFrame(0)
        f.error_code = errorCode
        f.last_stream_id = lastStreamID
        f.additional_data = additionalData
        return f

    def buildRstStreamFrame(self, streamID, errorCode=0):
        """
        Builds a single RST_STREAM frame.
        """
        f = hyperframe.frame.RstStreamFrame(streamID)
        f.error_code = errorCode
        return f

    def buildPriorityFrame(self, streamID, weight, dependsOn=0, exclusive=False):
        """
        Builds a single priority frame.
        """
        f = hyperframe.frame.PriorityFrame(streamID)
        f.depends_on = dependsOn
        f.stream_weight = weight
        f.exclusive = exclusive
        return f

    def buildPushPromiseFrame(self, streamID, promisedStreamID, headers, flags=[]):
        """
        Builds a single Push Promise frame.
        """
        f = hyperframe.frame.PushPromiseFrame(streamID)
        f.promised_stream_id = promisedStreamID
        f.data = self.encoder.encode(headers)
        f.flags = set(flags)
        f.flags.add('END_HEADERS')
        return f