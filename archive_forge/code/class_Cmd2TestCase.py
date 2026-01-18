import re
import unittest
from typing import (
from . import (
class Cmd2TestCase(unittest.TestCase):
    """A unittest class used for transcript testing.

    Subclass this, setting CmdApp, to make a unittest.TestCase class
    that will execute the commands in a transcript file and expect the
    results shown.

    See example.py
    """
    cmdapp: Optional['Cmd'] = None

    def setUp(self) -> None:
        if self.cmdapp:
            self._fetchTranscripts()
            self._orig_stdout = self.cmdapp.stdout
            self.cmdapp.stdout = cast(TextIO, utils.StdSim(cast(TextIO, self.cmdapp.stdout)))

    def tearDown(self) -> None:
        if self.cmdapp:
            self.cmdapp.stdout = self._orig_stdout

    def runTest(self) -> None:
        if self.cmdapp:
            its = sorted(self.transcripts.items())
            for fname, transcript in its:
                self._test_transcript(fname, transcript)

    def _fetchTranscripts(self) -> None:
        self.transcripts = {}
        testfiles = cast(List[str], getattr(self.cmdapp, 'testfiles', []))
        for fname in testfiles:
            tfile = open(fname)
            self.transcripts[fname] = iter(tfile.readlines())
            tfile.close()

    def _test_transcript(self, fname: str, transcript: Iterator[str]) -> None:
        if self.cmdapp is None:
            return
        line_num = 0
        finished = False
        line = ansi.strip_style(next(transcript))
        line_num += 1
        while not finished:
            while not line.startswith(self.cmdapp.visible_prompt):
                try:
                    line = ansi.strip_style(next(transcript))
                except StopIteration:
                    finished = True
                    break
                line_num += 1
            command_parts = [line[len(self.cmdapp.visible_prompt):]]
            try:
                line = next(transcript)
            except StopIteration:
                line = ''
            line_num += 1
            while line.startswith(self.cmdapp.continuation_prompt):
                command_parts.append(line[len(self.cmdapp.continuation_prompt):])
                try:
                    line = next(transcript)
                except StopIteration as exc:
                    msg = f'Transcript broke off while reading command beginning at line {line_num} with\n{command_parts[0]}'
                    raise StopIteration(msg) from exc
                line_num += 1
            command = ''.join(command_parts)
            stop = self.cmdapp.onecmd_plus_hooks(command)
            result = self.cmdapp.stdout.read()
            stop_msg = 'Command indicated application should quit, but more commands in transcript'
            if ansi.strip_style(line).startswith(self.cmdapp.visible_prompt):
                message = f'\nFile {fname}, line {line_num}\nCommand was:\n{command}\nExpected: (nothing)\nGot:\n{result}\n'
                self.assertTrue(not result.strip(), message)
                self.assertFalse(stop, stop_msg)
                continue
            expected_parts = []
            while not ansi.strip_style(line).startswith(self.cmdapp.visible_prompt):
                expected_parts.append(line)
                try:
                    line = next(transcript)
                except StopIteration:
                    finished = True
                    break
                line_num += 1
            if stop:
                self.assertTrue(finished, stop_msg)
            expected = ''.join(expected_parts)
            expected = self._transform_transcript_expected(expected)
            message = f'\nFile {fname}, line {line_num}\nCommand was:\n{command}\nExpected:\n{expected}\nGot:\n{result}\n'
            self.assertTrue(re.match(expected, result, re.MULTILINE | re.DOTALL), message)

    def _transform_transcript_expected(self, s: str) -> str:
        """Parse the string with slashed regexes into a valid regex.

        Given a string like:

            Match a 10 digit phone number: /\\d{3}-\\d{3}-\\d{4}/

        Turn it into a valid regular expression which matches the literal text
        of the string and the regular expression. We have to remove the slashes
        because they differentiate between plain text and a regular expression.
        Unless the slashes are escaped, in which case they are interpreted as
        plain text, or there is only one slash, which is treated as plain text
        also.

        Check the tests in tests/test_transcript.py to see all the edge
        cases.
        """
        regex = ''
        start = 0
        while True:
            regex, first_slash_pos, start = self._escaped_find(regex, s, start, False)
            if first_slash_pos == -1:
                regex += re.escape(s[start:])
                break
            else:
                regex += re.escape(s[start:first_slash_pos])
                start = first_slash_pos + 1
                regex, second_slash_pos, start = self._escaped_find(regex, s, start, True)
                if second_slash_pos > 0:
                    regex += s[start:second_slash_pos]
                    start = second_slash_pos + 1
                else:
                    regex += re.escape(s[start - 1:])
                    break
        return regex

    @staticmethod
    def _escaped_find(regex: str, s: str, start: int, in_regex: bool) -> Tuple[str, int, int]:
        """Find the next slash in {s} after {start} that is not preceded by a backslash.

        If we find an escaped slash, add everything up to and including it to regex,
        updating {start}. {start} therefore serves two purposes, tells us where to start
        looking for the next thing, and also tells us where in {s} we have already
        added things to {regex}

        {in_regex} specifies whether we are currently searching in a regex, we behave
        differently if we are or if we aren't.
        """
        while True:
            pos = s.find('/', start)
            if pos == -1:
                break
            elif pos == 0:
                break
            elif s[pos - 1:pos] == '\\':
                if in_regex:
                    regex += s[start:pos - 1]
                    regex += s[pos]
                else:
                    regex += re.escape(s[start:pos - 1])
                    regex += re.escape(s[pos])
                start = pos + 1
            else:
                break
        return (regex, pos, start)