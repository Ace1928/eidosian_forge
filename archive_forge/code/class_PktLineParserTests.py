from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class PktLineParserTests(TestCase):

    def test_none(self):
        pktlines = []
        parser = PktLineParser(pktlines.append)
        parser.parse(b'0000')
        self.assertEqual(pktlines, [None])
        self.assertEqual(b'', parser.get_tail())

    def test_small_fragments(self):
        pktlines = []
        parser = PktLineParser(pktlines.append)
        parser.parse(b'00')
        parser.parse(b'05')
        parser.parse(b'z0000')
        self.assertEqual(pktlines, [b'z', None])
        self.assertEqual(b'', parser.get_tail())

    def test_multiple_packets(self):
        pktlines = []
        parser = PktLineParser(pktlines.append)
        parser.parse(b'0005z0006aba')
        self.assertEqual(pktlines, [b'z', b'ab'])
        self.assertEqual(b'a', parser.get_tail())