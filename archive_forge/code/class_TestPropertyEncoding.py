from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TestPropertyEncoding(TestCase):
    """Tests for how the bug URLs are encoded as revision properties."""

    def test_encoding_one(self):
        self.assertEqual('http://example.com/bugs/1 fixed', bugtracker.encode_fixes_bug_urls([('http://example.com/bugs/1', 'fixed')]))

    def test_encoding_zero(self):
        self.assertEqual('', bugtracker.encode_fixes_bug_urls([]))

    def test_encoding_two(self):
        self.assertEqual('http://example.com/bugs/1 fixed\nhttp://example.com/bugs/2 related', bugtracker.encode_fixes_bug_urls([('http://example.com/bugs/1', 'fixed'), ('http://example.com/bugs/2', 'related')]))

    def test_encoding_with_space(self):
        self.assertRaises(bugtracker.InvalidBugUrl, bugtracker.encode_fixes_bug_urls, [('http://example.com/bugs/ 1', 'fixed')])