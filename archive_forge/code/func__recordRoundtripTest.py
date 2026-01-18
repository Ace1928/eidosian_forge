import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def _recordRoundtripTest(self, record):
    """
        Assert that encoding C{record} and then decoding the resulting bytes
        creates a record which compares equal to C{record}.

        @type record: L{dns.IEncodable}
        @param record: A record instance to encode
        """
    stream = BytesIO()
    record.encode(stream)
    length = stream.tell()
    stream.seek(0, 0)
    replica = record.__class__()
    replica.decode(stream, length)
    self.assertEqual(record, replica)