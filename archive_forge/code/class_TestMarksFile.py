from .... import tests
from .. import marks_file
class TestMarksFile(tests.TestCaseWithTransport):

    def test_read(self):
        self.build_tree_contents([('marks', ':1 jelmer@jelmer-rev1\n:2 joe@example.com-rev2\n')])
        self.assertEqual({b'1': b'jelmer@jelmer-rev1', b'2': b'joe@example.com-rev2'}, marks_file.import_marks('marks'))

    def test_write(self):
        marks_file.export_marks('marks', {b'1': b'jelmer@jelmer-rev1', b'2': b'joe@example.com-rev2'})
        self.assertFileEqual(':1 jelmer@jelmer-rev1\n:2 joe@example.com-rev2\n', 'marks')