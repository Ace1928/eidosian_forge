import unittest
from oslo_config import iniparser
class ParserTestCase(unittest.TestCase):

    def setUp(self):
        self.parser = TestParser()

    def test_blank_line(self):
        lines = ['']
        self.parser.parse(lines)
        self.assertEqual({}, self.parser.values)

    def test_assignment_equal(self):
        lines = ['foo = bar']
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': ['bar']}}, self.parser.values)

    def test_assignment_colon(self):
        lines = ['foo: bar']
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': ['bar']}}, self.parser.values)

    def test_assignment_multiline(self):
        lines = ['foo = bar0', '  bar1']
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': ['bar0', 'bar1']}}, self.parser.values)

    def test_assignment_multline_empty(self):
        lines = ['foo = bar0', '', '  bar1']
        self.assertRaises(iniparser.ParseError, self.parser.parse, lines)

    def test_section_assignment(self):
        lines = ['[test]', 'foo = bar']
        self.parser.parse(lines)
        self.assertEqual({'test': {'foo': ['bar']}}, self.parser.values)

    def test_new_section(self):
        lines = ['[foo]']
        self.parser.parse(lines)
        self.assertEqual('foo', self.parser.section)

    def test_comment(self):
        lines = ['# foobar']
        self.parser.parse(lines)
        self.assertTrue(self.parser.comment_called)

    def test_empty_assignment(self):
        lines = ['foo = ']
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': ['']}}, self.parser.values)

    def test_assignment_space_single_quote(self):
        lines = ["foo = ' bar '"]
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': [' bar ']}}, self.parser.values)

    def test_assignment_space_double_quote(self):
        lines = ['foo = " bar "']
        self.parser.parse(lines)
        self.assertEqual({'': {'foo': [' bar ']}}, self.parser.values)