import argparse
import io
from unittest import mock
from cliff.formatters import shell
from cliff.tests import base
from cliff.tests import test_columns
class TestShellFormatter(base.TestBase):

    def test(self):
        sf = shell.ShellFormatter()
        c = ('a', 'b', 'c', 'd')
        d = ('A', 'B', 'C', '"escape me"')
        expected = 'a="A"\nb="B"\nd="\\"escape me\\""\n'
        output = io.StringIO()
        args = mock.Mock()
        args.variables = ['a', 'b', 'd']
        args.prefix = ''
        sf.emit_one(c, d, output, args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_args(self):
        sf = shell.ShellFormatter()
        c = ('a', 'b', 'c', 'd')
        d = ('A', 'B', 'C', '"escape me"')
        expected = 'Xd="\\"escape me\\""\n'
        output = io.StringIO()
        parser = argparse.ArgumentParser(description='Testing...')
        sf.add_argument_group(parser)
        parsed_args = parser.parse_args(['--variable', 'd', '--prefix', 'X'])
        sf.emit_one(c, d, output, parsed_args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_formattable_column(self):
        sf = shell.ShellFormatter()
        c = ('a', 'b', 'c')
        d = ('A', 'B', test_columns.FauxColumn(['the', 'value']))
        expected = '\n'.join(['a="A"', 'b="B"', 'c="[\'the\', \'value\']"\n'])
        output = io.StringIO()
        args = mock.Mock()
        args.variables = ['a', 'b', 'c']
        args.prefix = ''
        sf.emit_one(c, d, output, args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_non_string_values(self):
        sf = shell.ShellFormatter()
        c = ('a', 'b', 'c', 'd', 'e')
        d = (True, False, 100, '"esc"', str('"esc"'))
        expected = 'a="True"\nb="False"\nc="100"\nd="\\"esc\\""\ne="\\"esc\\""\n'
        output = io.StringIO()
        args = mock.Mock()
        args.variables = ['a', 'b', 'c', 'd', 'e']
        args.prefix = ''
        sf.emit_one(c, d, output, args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)

    def test_non_bash_friendly_values(self):
        sf = shell.ShellFormatter()
        c = ('a', 'foo-bar', 'provider:network_type')
        d = (True, 'baz', 'vxlan')
        expected = 'a="True"\nfoo_bar="baz"\nprovider_network_type="vxlan"\n'
        output = io.StringIO()
        args = mock.Mock()
        args.variables = ['a', 'foo-bar', 'provider:network_type']
        args.prefix = ''
        sf.emit_one(c, d, output, args)
        actual = output.getvalue()
        self.assertEqual(expected, actual)