from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TestPropertyDecoding(TestCase):
    """Tests for parsing bug revision properties."""

    def test_decoding_one(self):
        self.assertEqual([('http://example.com/bugs/1', 'fixed')], list(bugtracker.decode_bug_urls('http://example.com/bugs/1 fixed')))

    def test_decoding_zero(self):
        self.assertEqual([], list(bugtracker.decode_bug_urls('')))

    def test_decoding_two(self):
        self.assertEqual([('http://example.com/bugs/1', 'fixed'), ('http://example.com/bugs/2', 'related')], list(bugtracker.decode_bug_urls('http://example.com/bugs/1 fixed\nhttp://example.com/bugs/2 related')))

    def test_decoding_invalid(self):
        self.assertRaises(bugtracker.InvalidLineInBugsProperty, list, bugtracker.decode_bug_urls('http://example.com/bugs/ 1 fixed\n'))