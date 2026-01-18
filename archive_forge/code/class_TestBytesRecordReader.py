from io import BytesIO
from ... import tests
from .. import pack
class TestBytesRecordReader(tests.TestCase):
    """Tests for reading and validating Bytes records with
    BytesRecordReader.

    Like TestContainerReader, this explicitly tests the reading of format 1
    data.  If a new version of the format is added, then a separate set of
    tests for reading that format should be added.
    """

    def get_reader_for(self, data):
        stream = BytesIO(data)
        reader = pack.BytesRecordReader(stream)
        return reader

    def test_record_with_no_name(self):
        """Reading a Bytes record with no name returns an empty list of
        names.
        """
        reader = self.get_reader_for(b'5\n\naaaaa')
        names, get_bytes = reader.read()
        self.assertEqual([], names)
        self.assertEqual(b'aaaaa', get_bytes(None))

    def test_record_with_one_name(self):
        """Reading a Bytes record with one name returns a list of just that
        name.
        """
        reader = self.get_reader_for(b'5\nname1\n\naaaaa')
        names, get_bytes = reader.read()
        self.assertEqual([(b'name1',)], names)
        self.assertEqual(b'aaaaa', get_bytes(None))

    def test_record_with_two_names(self):
        """Reading a Bytes record with two names returns a list of both names.
        """
        reader = self.get_reader_for(b'5\nname1\nname2\n\naaaaa')
        names, get_bytes = reader.read()
        self.assertEqual([(b'name1',), (b'name2',)], names)
        self.assertEqual(b'aaaaa', get_bytes(None))

    def test_record_with_two_part_names(self):
        """Reading a Bytes record with a two_part name reads both."""
        reader = self.get_reader_for(b'5\nname1\x00name2\n\naaaaa')
        names, get_bytes = reader.read()
        self.assertEqual([(b'name1', b'name2')], names)
        self.assertEqual(b'aaaaa', get_bytes(None))

    def test_invalid_length(self):
        """If the length-prefix is not a number, parsing raises
        InvalidRecordError.
        """
        reader = self.get_reader_for(b'not a number\n')
        self.assertRaises(pack.InvalidRecordError, reader.read)

    def test_early_eof(self):
        """Tests for premature EOF occuring during parsing Bytes records with
        BytesRecordReader.

        A incomplete container might be interrupted at any point.  The
        BytesRecordReader needs to cope with the input stream running out no
        matter where it is in the parsing process.

        In all cases, UnexpectedEndOfContainerError should be raised.
        """
        complete_record = b'6\nname\n\nabcdef'
        for count in range(0, len(complete_record)):
            incomplete_record = complete_record[:count]
            reader = self.get_reader_for(incomplete_record)
            try:
                names, read_bytes = reader.read()
                read_bytes(None)
            except pack.UnexpectedEndOfContainerError:
                pass
            else:
                self.fail('UnexpectedEndOfContainerError not raised when parsing %r' % (incomplete_record,))

    def test_initial_eof(self):
        """EOF before any bytes read at all."""
        reader = self.get_reader_for(b'')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)

    def test_eof_after_length(self):
        """EOF after reading the length and before reading name(s)."""
        reader = self.get_reader_for(b'123\n')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)

    def test_eof_during_name(self):
        """EOF during reading a name."""
        reader = self.get_reader_for(b'123\nname')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)

    def test_read_invalid_name_whitespace(self):
        """Names must have no whitespace."""
        reader = self.get_reader_for(b'0\nbad name\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.read)
        reader = self.get_reader_for(b'0\nbad\tname\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.read)
        reader = self.get_reader_for(b'0\nbad\x0bname\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.read)

    def test_validate_whitespace_in_name(self):
        """Names must have no whitespace."""
        reader = self.get_reader_for(b'0\nbad name\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.validate)

    def test_validate_interrupted_prelude(self):
        """EOF during reading a record's prelude causes validate to fail."""
        reader = self.get_reader_for(b'')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_interrupted_body(self):
        """EOF during reading a record's body causes validate to fail."""
        reader = self.get_reader_for(b'1\n\n')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_unparseable_length(self):
        """An unparseable record length causes validate to fail."""
        reader = self.get_reader_for(b'\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.validate)

    def test_validate_undecodeable_name(self):
        """Names that aren't valid UTF-8 cause validate to fail."""
        reader = self.get_reader_for(b'0\n\xcc\n\n')
        self.assertRaises(pack.InvalidRecordError, reader.validate)

    def test_read_max_length(self):
        """If the max_length passed to the callable returned by read is not
        None, then no more than that many bytes will be read.
        """
        reader = self.get_reader_for(b'6\n\nabcdef')
        names, get_bytes = reader.read()
        self.assertEqual(b'abc', get_bytes(3))

    def test_read_no_max_length(self):
        """If the max_length passed to the callable returned by read is None,
        then all the bytes in the record will be read.
        """
        reader = self.get_reader_for(b'6\n\nabcdef')
        names, get_bytes = reader.read()
        self.assertEqual(b'abcdef', get_bytes(None))

    def test_repeated_read_calls(self):
        """Repeated calls to the callable returned from BytesRecordReader.read
        will not read beyond the end of the record.
        """
        reader = self.get_reader_for(b'6\n\nabcdefB3\nnext-record\nXXX')
        names, get_bytes = reader.read()
        self.assertEqual(b'abcdef', get_bytes(None))
        self.assertEqual(b'', get_bytes(None))
        self.assertEqual(b'', get_bytes(99))