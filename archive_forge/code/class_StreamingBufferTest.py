import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
class StreamingBufferTest(unittest.TestCase):

    def setUp(self):
        self.stream = compression.StreamingBuffer()

    def testSimpleStream(self):
        """Test simple stream operations.

        Test that the stream can be written to and read from. Also test that
        reading from the stream consumes the bytes.
        """
        self.assertEqual(self.stream.length, 0)
        self.stream.write(b'Sample data')
        self.assertEqual(self.stream.length, 11)
        data = self.stream.read(11)
        self.assertEqual(data, b'Sample data')
        self.assertEqual(self.stream.length, 0)

    def testPartialReads(self):
        """Test partial stream reads.

        Test that the stream can be read in chunks while perserving the
        consumption mechanics.
        """
        self.stream.write(b'Sample data')
        data = self.stream.read(6)
        self.assertEqual(data, b'Sample')
        self.assertEqual(self.stream.length, 5)
        data = self.stream.read(5)
        self.assertEqual(data, b' data')
        self.assertEqual(self.stream.length, 0)

    def testTooShort(self):
        """Test excessive stream reads.

        Test that more data can be requested from the stream than available
        without raising an exception.
        """
        self.stream.write(b'Sample')
        data = self.stream.read(100)
        self.assertEqual(data, b'Sample')
        self.assertEqual(self.stream.length, 0)