from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestEcho(script.TestCaseWithMemoryTransportAndScript):

    def test_echo_usage(self):
        story = '\n$ echo foo\n<bar\n'
        self.assertRaises(SyntaxError, self.run_script, story)

    def test_echo_input(self):
        self.assertRaises(SyntaxError, self.run_script, '\n            $ echo <foo\n            ')

    def test_echo_to_output(self):
        retcode, out, err = self.run_command(['echo'], None, '\n', None)
        self.assertEqual('\n', out)
        self.assertEqual(None, err)

    def test_echo_some_to_output(self):
        retcode, out, err = self.run_command(['echo', 'hello'], None, 'hello\n', None)
        self.assertEqual('hello\n', out)
        self.assertEqual(None, err)

    def test_echo_more_output(self):
        retcode, out, err = self.run_command(['echo', 'hello', 'happy', 'world'], None, 'hello happy world\n', None)
        self.assertEqual('hello happy world\n', out)
        self.assertEqual(None, err)

    def test_echo_appended(self):
        retcode, out, err = self.run_command(['echo', 'hello', '>file'], None, None, None)
        self.assertEqual(None, out)
        self.assertEqual(None, err)
        self.assertFileEqual(b'hello\n', 'file')
        retcode, out, err = self.run_command(['echo', 'happy', '>>file'], None, None, None)
        self.assertEqual(None, out)
        self.assertEqual(None, err)
        self.assertFileEqual(b'hello\nhappy\n', 'file')

    def test_empty_line_in_output_is_respected(self):
        self.run_script('\n            $ echo\n\n            $ echo bar\n            bar\n            ')