from io import BytesIO
from ... import tests
from .. import pack
class TestContainerPushParser(PushParserTestCase):
    """Tests for ContainerPushParser.

    The ContainerPushParser reads format 1 containers, so these tests
    explicitly test how it reacts to format 1 data.  If a new version of the
    format is added, then separate tests for that format should be added.
    """

    def test_construct(self):
        """ContainerPushParser can be constructed."""
        pack.ContainerPushParser()

    def test_multiple_records_at_once(self):
        """If multiple records worth of data are fed to the parser in one
        string, the parser will correctly parse all the records.

        (A naive implementation might stop after parsing the first record.)
        """
        parser = self.make_parser_expecting_record_type()
        parser.accept_bytes(b'B5\nname1\n\nbody1B5\nname2\n\nbody2')
        self.assertEqual([([(b'name1',)], b'body1'), ([(b'name2',)], b'body2')], parser.read_pending_records())

    def test_multiple_empty_records_at_once(self):
        """If multiple empty records worth of data are fed to the parser in one
        string, the parser will correctly parse all the records.

        (A naive implementation might stop after parsing the first empty
        record, because the buffer size had not changed.)
        """
        parser = self.make_parser_expecting_record_type()
        parser.accept_bytes(b'B0\nname1\n\nB0\nname2\n\n')
        self.assertEqual([([(b'name1',)], b''), ([(b'name2',)], b'')], parser.read_pending_records())