from io import BytesIO
from ... import tests
from .. import pack
class TestContainerPushParserBytesParsing(PushParserTestCase):
    """Tests for reading Bytes records with ContainerPushParser.

    The ContainerPushParser reads format 1 containers, so these tests
    explicitly test how it reacts to format 1 data.  If a new version of the
    format is added, then separate tests for that format should be added.
    """

    def test_record_with_no_name(self):
        """Reading a Bytes record with no name returns an empty list of
        names.
        """
        self.assertRecordParsing(([], b'aaaaa'), b'5\n\naaaaa')

    def test_record_with_one_name(self):
        """Reading a Bytes record with one name returns a list of just that
        name.
        """
        self.assertRecordParsing(([(b'name1',)], b'aaaaa'), b'5\nname1\n\naaaaa')

    def test_record_with_two_names(self):
        """Reading a Bytes record with two names returns a list of both names.
        """
        self.assertRecordParsing(([(b'name1',), (b'name2',)], b'aaaaa'), b'5\nname1\nname2\n\naaaaa')

    def test_record_with_two_part_names(self):
        """Reading a Bytes record with a two_part name reads both."""
        self.assertRecordParsing(([(b'name1', b'name2')], b'aaaaa'), b'5\nname1\x00name2\n\naaaaa')

    def test_invalid_length(self):
        """If the length-prefix is not a number, parsing raises
        InvalidRecordError.
        """
        parser = self.make_parser_expecting_bytes_record()
        self.assertRaises(pack.InvalidRecordError, parser.accept_bytes, b'not a number\n')

    def test_incomplete_record(self):
        """If the bytes seen so far don't form a complete record, then there
        will be nothing returned by read_pending_records.
        """
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes(b'5\n\nabcd')
        self.assertEqual([], parser.read_pending_records())

    def test_accept_nothing(self):
        """The edge case of parsing an empty string causes no error."""
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes(b'')

    def assertInvalidRecord(self, data):
        """Assert that parsing the given bytes raises InvalidRecordError."""
        parser = self.make_parser_expecting_bytes_record()
        self.assertRaises(pack.InvalidRecordError, parser.accept_bytes, data)

    def test_read_invalid_name_whitespace(self):
        """Names must have no whitespace."""
        self.assertInvalidRecord(b'0\nbad name\n\n')
        self.assertInvalidRecord(b'0\nbad\tname\n\n')
        self.assertInvalidRecord(b'0\nbad\x0bname\n\n')

    def test_repeated_read_pending_records(self):
        """read_pending_records will not return the same record twice."""
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes(b'6\n\nabcdef')
        self.assertEqual([([], b'abcdef')], parser.read_pending_records())
        self.assertEqual([], parser.read_pending_records())