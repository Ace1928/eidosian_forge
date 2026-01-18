from .. import cmdline, tests
from .features import backslashdir_feature
class TestSplitter(tests.TestCase):

    def assertAsTokens(self, expected, line, single_quotes_allowed=False):
        s = cmdline.Splitter(line, single_quotes_allowed=single_quotes_allowed)
        self.assertEqual(expected, list(s))

    def test_simple(self):
        self.assertAsTokens([(False, 'foo'), (False, 'bar'), (False, 'baz')], 'foo bar baz')

    def test_ignore_multiple_spaces(self):
        self.assertAsTokens([(False, 'foo'), (False, 'bar')], 'foo  bar')

    def test_ignore_leading_space(self):
        self.assertAsTokens([(False, 'foo'), (False, 'bar')], '  foo bar')

    def test_ignore_trailing_space(self):
        self.assertAsTokens([(False, 'foo'), (False, 'bar')], 'foo bar  ')

    def test_posix_quotations(self):
        self.assertAsTokens([(True, 'foo bar')], "'foo bar'", single_quotes_allowed=True)
        self.assertAsTokens([(True, 'foo bar')], "'fo''o b''ar'", single_quotes_allowed=True)
        self.assertAsTokens([(True, 'foo bar')], '"fo""o b""ar"', single_quotes_allowed=True)
        self.assertAsTokens([(True, 'foo bar')], '"fo"\'o b\'"ar"', single_quotes_allowed=True)

    def test_nested_quotations(self):
        self.assertAsTokens([(True, 'foo"" bar')], '"foo\\"\\" bar"')
        self.assertAsTokens([(True, "foo'' bar")], '"foo\'\' bar"')
        self.assertAsTokens([(True, "foo'' bar")], '"foo\'\' bar"', single_quotes_allowed=True)
        self.assertAsTokens([(True, 'foo"" bar')], '\'foo"" bar\'', single_quotes_allowed=True)

    def test_empty_result(self):
        self.assertAsTokens([], '')
        self.assertAsTokens([], '    ')

    def test_quoted_empty(self):
        self.assertAsTokens([(True, '')], '""')
        self.assertAsTokens([(False, "''")], "''")
        self.assertAsTokens([(True, '')], "''", single_quotes_allowed=True)
        self.assertAsTokens([(False, 'a'), (True, ''), (False, 'c')], 'a "" c')
        self.assertAsTokens([(False, 'a'), (True, ''), (False, 'c')], "a '' c", single_quotes_allowed=True)

    def test_unicode_chars(self):
        self.assertAsTokens([(False, 'fµî'), (False, 'ሴ㑖')], 'fµî ሴ㑖')

    def test_newline_in_quoted_section(self):
        self.assertAsTokens([(True, 'foo\nbar\nbaz\n')], '"foo\nbar\nbaz\n"')
        self.assertAsTokens([(True, 'foo\nbar\nbaz\n')], "'foo\nbar\nbaz\n'", single_quotes_allowed=True)

    def test_escape_chars(self):
        self.assertAsTokens([(False, 'foo\\bar')], 'foo\\bar')

    def test_escape_quote(self):
        self.assertAsTokens([(True, 'foo"bar')], '"foo\\"bar"')
        self.assertAsTokens([(True, 'foo\\"bar')], '"foo\\\\\\"bar"')
        self.assertAsTokens([(True, 'foo\\bar')], '"foo\\\\"bar"')

    def test_double_escape(self):
        self.assertAsTokens([(True, 'foo\\\\bar')], '"foo\\\\bar"')
        self.assertAsTokens([(False, 'foo\\\\bar')], 'foo\\\\bar')

    def test_multiple_quoted_args(self):
        self.assertAsTokens([(True, 'x x'), (True, 'y y')], '"x x" "y y"')
        self.assertAsTokens([(True, 'x x'), (True, 'y y')], '"x x" \'y y\'', single_quotes_allowed=True)

    def test_n_backslashes_handling(self):
        self.requireFeature(backslashdir_feature)
        self.assertAsTokens([(True, '\\\\host\\path')], '"\\\\host\\path"')
        self.assertAsTokens([(False, '\\\\host\\path')], '\\\\host\\path')
        self.assertAsTokens([(True, '\\\\'), (False, '*.py')], '"\\\\\\\\" *.py')
        self.assertAsTokens([(True, '\\\\" *.py')], '"\\\\\\\\\\" *.py"')
        self.assertAsTokens([(True, '\\\\ *.py')], '\\\\\\\\" *.py"')
        self.assertAsTokens([(False, '\\\\"'), (False, '*.py')], '\\\\\\\\\\" *.py')
        self.assertAsTokens([(True, '\\\\')], '"\\\\')