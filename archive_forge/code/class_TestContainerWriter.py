from io import BytesIO
from ... import tests
from .. import pack
class TestContainerWriter(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.output = BytesIO()
        self.writer = pack.ContainerWriter(self.output.write)

    def assertOutput(self, expected_output):
        """Assert that the output of self.writer ContainerWriter is equal to
        expected_output.
        """
        self.assertEqual(expected_output, self.output.getvalue())

    def test_construct(self):
        """Test constructing a ContainerWriter.

        This uses None as the output stream to show that the constructor
        doesn't try to use the output stream.
        """
        pack.ContainerWriter(None)

    def test_begin(self):
        """The begin() method writes the container format marker line."""
        self.writer.begin()
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\n')

    def test_zero_records_written_after_begin(self):
        """After begin is written, 0 records have been written."""
        self.writer.begin()
        self.assertEqual(0, self.writer.records_written)

    def test_end(self):
        """The end() method writes an End Marker record."""
        self.writer.begin()
        self.writer.end()
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nE')

    def test_empty_end_does_not_add_a_record_to_records_written(self):
        """The end() method does not count towards the records written."""
        self.writer.begin()
        self.writer.end()
        self.assertEqual(0, self.writer.records_written)

    def test_non_empty_end_does_not_add_a_record_to_records_written(self):
        """The end() method does not count towards the records written."""
        self.writer.begin()
        self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
        self.writer.end()
        self.assertEqual(1, self.writer.records_written)

    def test_add_bytes_record_no_name(self):
        """Add a bytes record with no name."""
        self.writer.begin()
        offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[])
        self.assertEqual((42, 7), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabc')

    def test_add_bytes_record_one_name(self):
        """Add a bytes record with one name."""
        self.writer.begin()
        offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[(b'name1',)])
        self.assertEqual((42, 13), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\n\nabc')

    def test_add_bytes_record_split_writes(self):
        """Write a large record which does multiple IOs"""
        writes = []
        real_write = self.writer.write_func

        def record_writes(data):
            writes.append(data)
            return real_write(data)
        self.writer.write_func = record_writes
        self.writer._JOIN_WRITES_THRESHOLD = 2
        self.writer.begin()
        offset, length = self.writer.add_bytes_record([b'abcabc'], len(b'abcabc'), names=[(b'name1',)])
        self.assertEqual((42, 16), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB6\nname1\n\nabcabc')
        self.assertEqual([b'Bazaar pack format 1 (introduced in 0.18)\n', b'B6\nname1\n\n', b'abcabc'], writes)

    def test_add_bytes_record_two_names(self):
        """Add a bytes record with two names."""
        self.writer.begin()
        offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[(b'name1',), (b'name2',)])
        self.assertEqual((42, 19), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\nname2\n\nabc')

    def test_add_bytes_record_two_element_name(self):
        """Add a bytes record with a two-element name."""
        self.writer.begin()
        offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[(b'name1', b'name2')])
        self.assertEqual((42, 19), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\x00name2\n\nabc')

    def test_add_second_bytes_record_gets_higher_offset(self):
        self.writer.begin()
        self.writer.add_bytes_record([b'a', b'bc'], len(b'abc'), names=[])
        offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[])
        self.assertEqual((49, 7), (offset, length))
        self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabcB3\n\nabc')

    def test_add_bytes_record_invalid_name(self):
        """Adding a Bytes record with a name with whitespace in it raises
        InvalidRecordError.
        """
        self.writer.begin()
        self.assertRaises(pack.InvalidRecordError, self.writer.add_bytes_record, [b'abc'], len(b'abc'), names=[(b'bad name',)])

    def test_add_bytes_records_add_to_records_written(self):
        """Adding a Bytes record increments the records_written counter."""
        self.writer.begin()
        self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
        self.assertEqual(1, self.writer.records_written)
        self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
        self.assertEqual(2, self.writer.records_written)