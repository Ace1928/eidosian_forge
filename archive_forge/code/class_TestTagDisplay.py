from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestTagDisplay(TestCase):

    def test_tag(self):
        tagger = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.TagCommand(b'refs/tags/v1.0', b':xxx', tagger, b'create v1.0')
        self.assertEqual(b'tag refs/tags/v1.0\nfrom :xxx\ntagger Joe Wong <joe@example.com> 1234567890 -0600\ndata 11\ncreate v1.0', bytes(c))

    def test_tag_no_from(self):
        tagger = (b'Joe Wong', b'joe@example.com', 1234567890, -6 * 3600)
        c = commands.TagCommand(b'refs/tags/v1.0', None, tagger, b'create v1.0')
        self.assertEqual(b'tag refs/tags/v1.0\ntagger Joe Wong <joe@example.com> 1234567890 -0600\ndata 11\ncreate v1.0', bytes(c))