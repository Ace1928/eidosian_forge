from ...revision import Revision
from ..chk_serializer import chk_bencode_serializer
from . import TestCase
class TestBEncodeSerializer1(TestCase):
    """Test BEncode serialization"""

    def test_unpack_revision(self):
        """Test unpacking a revision"""
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1)
        self.assertEqual(rev.committer, 'Canonical.com Patch Queue Manager <pqm@pqm.ubuntu.com>')
        self.assertEqual(rev.inventory_sha1, b'4a2c7fb50e077699242cf6eb16a61779c7b680a7')
        self.assertEqual([b'pqm@pqm.ubuntu.com-20090514104039-kggemn7lrretzpvc', b'jelmer@samba.org-20090510012654-jp9ufxquekaokbeo'], rev.parent_ids)
        self.assertEqual('(Jelmer) Move dpush to InterBranch.', rev.message)
        self.assertEqual(b'pqm@pqm.ubuntu.com-20090514113250-jntkkpminfn3e0tz', rev.revision_id)
        self.assertEqual({'branch-nick': '+trunk'}, rev.properties)
        self.assertEqual(3600, rev.timezone)

    def test_written_form_matches(self):
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1)
        as_str = chk_bencode_serializer.write_revision_to_string(rev)
        self.assertEqualDiff(_working_revision_bencode1, as_str)

    def test_unpack_revision_no_timezone(self):
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1_no_timezone)
        self.assertEqual(None, rev.timezone)

    def assertRoundTrips(self, serializer, orig_rev):
        lines = serializer.write_revision_to_lines(orig_rev)
        new_rev = serializer.read_revision_from_string(b''.join(lines))
        self.assertEqual(orig_rev, new_rev)

    def test_roundtrips_non_ascii(self):
        rev = Revision(b'revid1')
        rev.message = '\nåme'
        rev.committer = 'Erik Bågfors'
        rev.timestamp = 1242385452
        rev.inventory_sha1 = b'4a2c7fb50e077699242cf6eb16a61779c7b680a7'
        rev.timezone = 3600
        self.assertRoundTrips(chk_bencode_serializer, rev)

    def test_roundtrips_xml_invalid_chars(self):
        rev = Revision(b'revid1')
        rev.message = '\t\ue000'
        rev.committer = 'Erik Bågfors'
        rev.timestamp = 1242385452
        rev.timezone = 3600
        rev.inventory_sha1 = b'4a2c7fb50e077699242cf6eb16a61779c7b680a7'
        self.assertRoundTrips(chk_bencode_serializer, rev)