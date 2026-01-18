import unittest
from testtools.compat import _b
from subunit import content, content_type, details
class TestSimpleDetails(unittest.TestCase):

    def test_lineReceived(self):
        parser = details.SimpleDetailsParser(None)
        parser.lineReceived(_b('foo\n'))
        parser.lineReceived(_b('bar\n'))
        self.assertEqual(_b('foo\nbar\n'), parser._message)

    def test_lineReceived_escaped_bracket(self):
        parser = details.SimpleDetailsParser(None)
        parser.lineReceived(_b('foo\n'))
        parser.lineReceived(_b(' ]are\n'))
        parser.lineReceived(_b('bar\n'))
        self.assertEqual(_b('foo\n]are\nbar\n'), parser._message)

    def test_get_message(self):
        parser = details.SimpleDetailsParser(None)
        self.assertEqual(_b(''), parser.get_message())

    def test_get_details(self):
        parser = details.SimpleDetailsParser(None)
        expected = {}
        expected['traceback'] = content.Content(content_type.ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b('')])
        found = parser.get_details()
        self.assertEqual(expected.keys(), found.keys())
        self.assertEqual(expected['traceback'].content_type, found['traceback'].content_type)
        self.assertEqual(_b('').join(expected['traceback'].iter_bytes()), _b('').join(found['traceback'].iter_bytes()))

    def test_get_details_skip(self):
        parser = details.SimpleDetailsParser(None)
        expected = {}
        expected['reason'] = content.Content(content_type.ContentType('text', 'plain'), lambda: [_b('')])
        found = parser.get_details('skip')
        self.assertEqual(expected, found)

    def test_get_details_success(self):
        parser = details.SimpleDetailsParser(None)
        expected = {}
        expected['message'] = content.Content(content_type.ContentType('text', 'plain'), lambda: [_b('')])
        found = parser.get_details('success')
        self.assertEqual(expected, found)