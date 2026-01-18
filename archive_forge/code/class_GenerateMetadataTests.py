from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
class GenerateMetadataTests(TestCase):

    def test_roundtrip(self):
        metadata = CommitSupplement()
        metadata.revision_id = b'myrevid'
        msg = inject_bzr_metadata(b'Foo', metadata, 'utf-8')
        self.assertEqual(b'Foo\n--BZR--\nrevision-id: myrevid\n', msg)

    def test_no_metadata(self):
        metadata = CommitSupplement()
        msg = inject_bzr_metadata(b'Foo', metadata, 'utf-8')
        self.assertEqual(b'Foo', msg)