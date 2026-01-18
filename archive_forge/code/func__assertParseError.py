import unittest
from oslo_config import iniparser
def _assertParseError(self, *lines):
    self.assertRaises(iniparser.ParseError, self.parser.parse, lines)