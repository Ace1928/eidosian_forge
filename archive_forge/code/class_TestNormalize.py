import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestNormalize(tests.TestCase):

    def test_single_line(self):
        s = 'foobar'
        e = '"foobar"'
        self.assertEqual(export_pot._normalize(s), e)
        s = 'foo"bar'
        e = '"foo\\"bar"'
        self.assertEqual(export_pot._normalize(s), e)

    def test_multi_lines(self):
        s = 'foo\nbar\n'
        e = '""\n"foo\\n"\n"bar\\n"'
        self.assertEqual(export_pot._normalize(s), e)
        s = '\nfoo\nbar\n'
        e = '""\n"\\n"\n"foo\\n"\n"bar\\n"'
        self.assertEqual(export_pot._normalize(s), e)