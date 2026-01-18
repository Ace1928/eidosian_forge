import re
import unittest
from typing import (
from . import (
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