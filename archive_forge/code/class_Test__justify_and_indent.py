import sys
import unittest
class Test__justify_and_indent(unittest.TestCase):

    def _callFUT(self, text, level, **kw):
        from zope.interface.document import _justify_and_indent
        return _justify_and_indent(text, level, **kw)

    def test_simple_level_0(self):
        LINES = ['Three blind mice', 'See how they run']
        text = '\n'.join(LINES)
        self.assertEqual(self._callFUT(text, 0), text)

    def test_simple_level_1(self):
        LINES = ['Three blind mice', 'See how they run']
        text = '\n'.join(LINES)
        self.assertEqual(self._callFUT(text, 1), '\n'.join([' ' + line for line in LINES]))

    def test_simple_level_2(self):
        LINES = ['Three blind mice', 'See how they run']
        text = '\n'.join(LINES)
        self.assertEqual(self._callFUT(text, 1), '\n'.join([' ' + line for line in LINES]))

    def test_simple_w_CRLF(self):
        LINES = ['Three blind mice', 'See how they run']
        text = '\r\n'.join(LINES)
        self.assertEqual(self._callFUT(text, 1), '\n'.join([' ' + line for line in LINES]))

    def test_with_munge(self):
        TEXT = 'This is a piece of text longer than 15 characters, \nand split across multiple lines.'
        EXPECTED = '  This is a piece\n  of text longer\n  than 15 characters,\n  and split across\n  multiple lines.\n '
        self.assertEqual(self._callFUT(TEXT, 1, munge=1, width=15), EXPECTED)