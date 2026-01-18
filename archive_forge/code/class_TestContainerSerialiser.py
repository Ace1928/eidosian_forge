from io import BytesIO
from ... import tests
from .. import pack
class TestContainerSerialiser(tests.TestCase):
    """Tests for the ContainerSerialiser class."""

    def test_construct(self):
        """Test constructing a ContainerSerialiser."""
        pack.ContainerSerialiser()

    def test_begin(self):
        serialiser = pack.ContainerSerialiser()
        self.assertEqual(b'Bazaar pack format 1 (introduced in 0.18)\n', serialiser.begin())

    def test_end(self):
        serialiser = pack.ContainerSerialiser()
        self.assertEqual(b'E', serialiser.end())

    def test_bytes_record_no_name(self):
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record(b'bytes', [])
        self.assertEqual(b'B5\n\nbytes', record)

    def test_bytes_record_one_name_with_one_part(self):
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record(b'bytes', [(b'name',)])
        self.assertEqual(b'B5\nname\n\nbytes', record)

    def test_bytes_record_one_name_with_two_parts(self):
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record(b'bytes', [(b'part1', b'part2')])
        self.assertEqual(b'B5\npart1\x00part2\n\nbytes', record)

    def test_bytes_record_two_names(self):
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record(b'bytes', [(b'name1',), (b'name2',)])
        self.assertEqual(b'B5\nname1\nname2\n\nbytes', record)

    def test_bytes_record_whitespace_in_name_part(self):
        serialiser = pack.ContainerSerialiser()
        self.assertRaises(pack.InvalidRecordError, serialiser.bytes_record, b'bytes', [(b'bad name',)])

    def test_bytes_record_header(self):
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_header(32, [(b'name1',), (b'name2',)])
        self.assertEqual(b'B32\nname1\nname2\n\n', record)