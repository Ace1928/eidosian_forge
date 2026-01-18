from io import BytesIO
from ... import tests
from .. import pack
class TestContainerReader(tests.TestCase):
    """Tests for the ContainerReader.

    The ContainerReader reads format 1 containers, so these tests explicitly
    test how it reacts to format 1 data.  If a new version of the format is
    added, then separate tests for that format should be added.
    """

    def get_reader_for(self, data):
        stream = BytesIO(data)
        reader = pack.ContainerReader(stream)
        return reader

    def test_construct(self):
        """Test constructing a ContainerReader.

        This uses None as the output stream to show that the constructor
        doesn't try to use the input stream.
        """
        pack.ContainerReader(None)

    def test_empty_container(self):
        """Read an empty container."""
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nE')
        self.assertEqual([], list(reader.iter_records()))

    def test_unknown_format(self):
        """Unrecognised container formats raise UnknownContainerFormatError."""
        reader = self.get_reader_for(b'unknown format\n')
        self.assertRaises(pack.UnknownContainerFormatError, reader.iter_records)

    def test_unexpected_end_of_container(self):
        """Containers that don't end with an End Marker record should cause
        UnexpectedEndOfContainerError to be raised.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\n')
        iterator = reader.iter_records()
        self.assertRaises(pack.UnexpectedEndOfContainerError, next, iterator)

    def test_unknown_record_type(self):
        """Unknown record types cause UnknownRecordTypeError to be raised."""
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nX')
        iterator = reader.iter_records()
        self.assertRaises(pack.UnknownRecordTypeError, next, iterator)

    def test_container_with_one_unnamed_record(self):
        """Read a container with one Bytes record.

        Parsing Bytes records is more thoroughly exercised by
        TestBytesRecordReader.  This test is here to ensure that
        ContainerReader's integration with BytesRecordReader is working.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB5\n\naaaaaE')
        expected_records = [([], b'aaaaa')]
        self.assertEqual(expected_records, [(names, read_bytes(None)) for names, read_bytes in reader.iter_records()])

    def test_validate_empty_container(self):
        """validate does not raise an error for a container with no records."""
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nE')
        reader.validate()

    def test_validate_non_empty_valid_container(self):
        """validate does not raise an error for a container with a valid record.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname\n\nabcE')
        reader.validate()

    def test_validate_bad_format(self):
        """validate raises an error for unrecognised format strings.

        It may raise either UnexpectedEndOfContainerError or
        UnknownContainerFormatError, depending on exactly what the string is.
        """
        inputs = [b'', b'x', b'Bazaar pack format 1 (introduced in 0.18)', b'bad\n']
        for input in inputs:
            reader = self.get_reader_for(input)
            self.assertRaises((pack.UnexpectedEndOfContainerError, pack.UnknownContainerFormatError), reader.validate)

    def test_validate_bad_record_marker(self):
        """validate raises UnknownRecordTypeError for unrecognised record
        types.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nX')
        self.assertRaises(pack.UnknownRecordTypeError, reader.validate)

    def test_validate_data_after_end_marker(self):
        """validate raises ContainerHasExcessDataError if there are any bytes
        after the end of the container.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nEcrud')
        self.assertRaises(pack.ContainerHasExcessDataError, reader.validate)

    def test_validate_no_end_marker(self):
        """validate raises UnexpectedEndOfContainerError if there's no end of
        container marker, even if the container up to this point has been
        valid.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\n')
        self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_duplicate_name(self):
        """validate raises DuplicateRecordNameError if the same name occurs
        multiple times in the container.
        """
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB0\nname\n\nB0\nname\n\nE')
        self.assertRaises(pack.DuplicateRecordNameError, reader.validate)

    def test_validate_undecodeable_name(self):
        """Names that aren't valid UTF-8 cause validate to fail."""
        reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB0\n\xcc\n\nE')
        self.assertRaises(pack.InvalidRecordError, reader.validate)