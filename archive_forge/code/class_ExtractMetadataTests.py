from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
class ExtractMetadataTests(TestCase):

    def test_roundtrip(self):
        msg, metadata = extract_bzr_metadata(b'Foo\n--BZR--\nrevision-id: foo\n')
        self.assertEqual(b'Foo', msg)
        self.assertEqual(b'foo', metadata.revision_id)