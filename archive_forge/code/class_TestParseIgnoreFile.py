import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
class TestParseIgnoreFile(TestCase):

    def test_parse_fancy(self):
        ignored = ignores.parse_ignore_file(BytesIO(b'./rootdir\nrandomfile*\npath/from/ro?t\nunicode\xc2\xb5\ndos\r\n\n#comment\n xx \n!RE:^\\.z.*\n!!./.zcompdump\n'))
        self.assertEqual({'./rootdir', 'randomfile*', 'path/from/ro?t', 'unicodeµ', 'dos', ' xx ', '!RE:^\\.z.*', '!!./.zcompdump'}, ignored)

    def test_parse_empty(self):
        ignored = ignores.parse_ignore_file(BytesIO(b''))
        self.assertEqual(set(), ignored)

    def test_parse_non_utf8(self):
        """Lines with non utf 8 characters should be discarded."""
        ignored = ignores.parse_ignore_file(BytesIO(b'utf8filename_a\ninvalid utf8\x80\nutf8filename_b\n'))
        self.assertEqual({'utf8filename_a', 'utf8filename_b'}, ignored)