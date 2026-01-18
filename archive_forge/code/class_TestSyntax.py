from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestSyntax(tests.TestCase):

    def test_comment_is_ignored(self):
        self.assertEqual([], script._script_to_commands('#comment\n'))

    def test_comment_multiple_lines(self):
        self.assertEqual([(['bar'], None, None, None)], script._script_to_commands('\n            # this comment is ignored\n            # so is this\n            # no we run bar\n            $ bar\n            '))

    def test_trim_blank_lines(self):
        """Blank lines are respected, but trimmed at the start and end.

        Python triple-quoted syntax is going to give stubby/empty blank lines
        right at the start and the end.  These are cut off so that callers don't
        need special syntax to avoid them.

        However we do want to be able to match commands that emit blank lines.
        """
        self.assertEqual([(['bar'], None, '\n', None)], script._script_to_commands('\n            $bar\n\n            '))

    def test_simple_command(self):
        self.assertEqual([(['cd', 'trunk'], None, None, None)], script._script_to_commands('$ cd trunk'))

    def test_command_with_single_quoted_param(self):
        story = "$ brz commit -m 'two words'"
        self.assertEqual([(['brz', 'commit', '-m', "'two words'"], None, None, None)], script._script_to_commands(story))

    def test_command_with_double_quoted_param(self):
        story = '$ brz commit -m "two words" '
        self.assertEqual([(['brz', 'commit', '-m', '"two words"'], None, None, None)], script._script_to_commands(story))

    def test_command_with_input(self):
        self.assertEqual([(['cat', '>file'], 'content\n', None, None)], script._script_to_commands('$ cat >file\n<content\n'))

    def test_indented(self):
        story = '\n            $ brz add\n            adding file\n            adding file2\n            '
        self.assertEqual([(['brz', 'add'], None, 'adding file\nadding file2\n', None)], script._script_to_commands(story))

    def test_command_with_output(self):
        story = '\n$ brz add\nadding file\nadding file2\n'
        self.assertEqual([(['brz', 'add'], None, 'adding file\nadding file2\n', None)], script._script_to_commands(story))

    def test_command_with_error(self):
        story = '\n$ brz branch foo\n2>brz: ERROR: Not a branch: "foo"\n'
        self.assertEqual([(['brz', 'branch', 'foo'], None, None, 'brz: ERROR: Not a branch: "foo"\n')], script._script_to_commands(story))

    def test_input_without_command(self):
        self.assertRaises(SyntaxError, script._script_to_commands, '<input')

    def test_output_without_command(self):
        self.assertRaises(SyntaxError, script._script_to_commands, '>input')

    def test_command_with_backquotes(self):
        story = '\n$ foo = `brz file-id toto`\n'
        self.assertEqual([(['foo', '=', '`brz file-id toto`'], None, None, None)], script._script_to_commands(story))