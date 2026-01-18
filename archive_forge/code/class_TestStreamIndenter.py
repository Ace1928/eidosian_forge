from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
class TestStreamIndenter(unittest.TestCase):

    def test_noprefix(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.write('Hello?\nHello, world!')
        self.assertEqual('    Hello?\n    Hello, world!', OUT2.getvalue())

    def test_prefix(self):
        prefix = 'foo:'
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1, prefix)
        OUT2.write('Hello?\nHello, world!')
        self.assertEqual('foo:Hello?\nfoo:Hello, world!', OUT2.getvalue())

    def test_blank_lines(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.write('Hello?\n\nText\n\nHello, world!')
        self.assertEqual('    Hello?\n\n    Text\n\n    Hello, world!', OUT2.getvalue())

    def test_writelines(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.writelines(['Hello?\n', '\n', 'Text\n', '\n', 'Hello, world!'])
        self.assertEqual('    Hello?\n\n    Text\n\n    Hello, world!', OUT2.getvalue())