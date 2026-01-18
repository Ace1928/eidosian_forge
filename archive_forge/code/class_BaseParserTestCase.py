import unittest
from oslo_config import iniparser
class BaseParserTestCase(unittest.TestCase):

    def setUp(self):
        self.parser = iniparser.BaseParser()

    def _assertParseError(self, *lines):
        self.assertRaises(iniparser.ParseError, self.parser.parse, lines)

    def test_invalid_assignment(self):
        self._assertParseError('foo - bar')

    def test_empty_key(self):
        self._assertParseError(': bar')

    def test_unexpected_continuation(self):
        self._assertParseError('   baz')

    def test_invalid_section(self):
        self._assertParseError('[section')

    def test_no_section_name(self):
        self._assertParseError('[]')