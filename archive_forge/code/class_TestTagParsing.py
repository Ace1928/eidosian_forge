import io
import time
import unittest
from fastimport import (
from :2
class TestTagParsing(unittest.TestCase):

    def test_tagger_with_email(self):
        p = parser.ImportParser(io.BytesIO(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong <joe@example.com> 1234567890 -0600\ndata 11\ncreate v1.0'))
        cmds = list(p.iter_commands())
        self.assertEqual(1, len(cmds))
        self.assertTrue(isinstance(cmds[0], commands.TagCommand))
        self.assertEqual(cmds[0].tagger, (b'Joe Wong', b'joe@example.com', 1234567890.0, -21600))

    def test_tagger_no_email_strict(self):
        p = parser.ImportParser(io.BytesIO(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong\ndata 11\ncreate v1.0'))
        self.assertRaises(errors.BadFormat, list, p.iter_commands())

    def test_tagger_no_email_not_strict(self):
        p = parser.ImportParser(io.BytesIO(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong\ndata 11\ncreate v1.0'), strict=False)
        cmds = list(p.iter_commands())
        self.assertEqual(1, len(cmds))
        self.assertTrue(isinstance(cmds[0], commands.TagCommand))
        self.assertEqual(cmds[0].tagger[:2], (b'Joe Wong', None))