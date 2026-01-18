from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class CapabilitiesTestCase(TestCase):

    def test_plain(self):
        self.assertEqual((b'bla', []), extract_capabilities(b'bla'))

    def test_caps(self):
        self.assertEqual((b'bla', [b'la']), extract_capabilities(b'bla\x00la'))
        self.assertEqual((b'bla', [b'la']), extract_capabilities(b'bla\x00la\n'))
        self.assertEqual((b'bla', [b'la', b'la']), extract_capabilities(b'bla\x00la la'))

    def test_plain_want_line(self):
        self.assertEqual((b'want bla', []), extract_want_line_capabilities(b'want bla'))

    def test_caps_want_line(self):
        self.assertEqual((b'want bla', [b'la']), extract_want_line_capabilities(b'want bla la'))
        self.assertEqual((b'want bla', [b'la']), extract_want_line_capabilities(b'want bla la\n'))
        self.assertEqual((b'want bla', [b'la', b'la']), extract_want_line_capabilities(b'want bla la la'))

    def test_ack_type(self):
        self.assertEqual(SINGLE_ACK, ack_type([b'foo', b'bar']))
        self.assertEqual(MULTI_ACK, ack_type([b'foo', b'bar', b'multi_ack']))
        self.assertEqual(MULTI_ACK_DETAILED, ack_type([b'foo', b'bar', b'multi_ack_detailed']))
        self.assertEqual(MULTI_ACK_DETAILED, ack_type([b'foo', b'bar', b'multi_ack', b'multi_ack_detailed']))