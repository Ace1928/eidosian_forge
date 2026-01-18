import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestParseSource(tests.TestCase):
    """Check mappings to line numbers generated from python source"""

    def test_classes(self):
        src = '\nclass Ancient:\n    """Old style class"""\n\nclass Modern(object):\n    """New style class"""\n'
        cls_lines, _ = export_pot._parse_source(src)
        self.assertEqual(cls_lines, {'Ancient': 2, 'Modern': 5})

    def test_classes_nested(self):
        src = '\nclass Matroska(object):\n    class Smaller(object):\n        class Smallest(object):\n            pass\n'
        cls_lines, _ = export_pot._parse_source(src)
        self.assertEqual(cls_lines, {'Matroska': 2, 'Smaller': 3, 'Smallest': 4})

    def test_strings_docstrings(self):
        src = '"""Module"""\n\ndef function():\n    """Function"""\n\nclass Class(object):\n    """Class"""\n\n    def method(self):\n        """Method"""\n'
        _, str_lines = export_pot._parse_source(src)
        self.assertEqual(str_lines, {'Module': 1, 'Function': 4, 'Class': 7, 'Method': 10})

    def test_strings_literals(self):
        src = 's = "One"\nt = (2, "Two")\nf = dict(key="Three")\n'
        _, str_lines = export_pot._parse_source(src)
        self.assertEqual(str_lines, {'One': 1, 'Two': 2, 'Three': 3})

    def test_strings_multiline(self):
        src = '"""Start\n\nEnd\n"""\nt = (\n    "A"\n    "B"\n    "C"\n    )\n'
        _, str_lines = export_pot._parse_source(src)
        self.assertEqual(str_lines, {'Start\n\nEnd\n': 1, 'ABC': 6})

    def test_strings_multiline_escapes(self):
        src = 's = "Escaped\\n"\nr = r"Raw\\n"\nt = (\n    "A\\n\\n"\n    "B\\n\\n"\n    "C\\n\\n"\n    )\n'
        _, str_lines = export_pot._parse_source(src)
        if sys.version_info < (3, 8):
            self.expectFailure('Escaped newlines confuses the multiline handling', self.assertNotEqual, str_lines, {'Escaped\n': 0, 'Raw\\n': 2, 'A\n\nB\n\nC\n\n': -2})
        else:
            self.assertEqual(str_lines, {'Escaped\n': 1, 'Raw\\n': 2, 'A\n\nB\n\nC\n\n': 4})