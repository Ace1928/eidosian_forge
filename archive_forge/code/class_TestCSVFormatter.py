import argparse
import io
import unittest
from unittest import mock
from cliff.formatters import commaseparated
from cliff.tests import test_columns
class TestCSVFormatter(unittest.TestCase):

    def test_commaseparated_list_formatter(self):
        sf = commaseparated.CSVLister()
        c = ('a', 'b', 'c')
        d1 = ('A', 'B', 'C')
        d2 = ('D', 'E', 'F')
        data = [d1, d2]
        expected = 'a,b,c\nA,B,C\nD,E,F\n'
        output = io.StringIO()
        parsed_args = mock.Mock()
        parsed_args.quote_mode = 'none'
        sf.emit_list(c, data, output, parsed_args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_commaseparated_list_formatter_quoted(self):
        sf = commaseparated.CSVLister()
        c = ('a', 'b', 'c')
        d1 = ('A', 'B', 'C')
        d2 = ('D', 'E', 'F')
        data = [d1, d2]
        expected = '"a","b","c"\n"A","B","C"\n"D","E","F"\n'
        output = io.StringIO()
        parser = argparse.ArgumentParser(description='Testing...')
        sf.add_argument_group(parser)
        parsed_args = parser.parse_args(['--quote', 'all'])
        sf.emit_list(c, data, output, parsed_args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_commaseparated_list_formatter_formattable_column(self):
        sf = commaseparated.CSVLister()
        c = ('a', 'b', 'c')
        d1 = ('A', 'B', test_columns.FauxColumn(['the', 'value']))
        data = [d1]
        expected = "a,b,c\nA,B,['the'\\, 'value']\n"
        output = io.StringIO()
        parsed_args = mock.Mock()
        parsed_args.quote_mode = 'none'
        sf.emit_list(c, data, output, parsed_args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_commaseparated_list_formatter_unicode(self):
        sf = commaseparated.CSVLister()
        c = ('a', 'b', 'c')
        d1 = ('A', 'B', 'C')
        happy = '高兴'
        d2 = ('D', 'E', happy)
        data = [d1, d2]
        expected = 'a,b,c\nA,B,C\nD,E,%s\n' % happy
        output = io.StringIO()
        parsed_args = mock.Mock()
        parsed_args.quote_mode = 'none'
        sf.emit_list(c, data, output, parsed_args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)