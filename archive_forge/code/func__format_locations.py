import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
def _format_locations(self, frames_lines, *, has_introduction):
    prepend_with_new_line = has_introduction
    regex = '^  File "(?P<file>.*?)", line (?P<line>[^,]+)(?:, in (?P<function>.*))?\\n'
    for frame in frames_lines:
        match = re.match(regex, frame)
        if match:
            file, line, function = match.group('file', 'line', 'function')
            is_mine = self._is_file_mine(file)
            if function is not None:
                pattern = '  File "{}", line {}, in {}\n'
            else:
                pattern = '  File "{}", line {}\n'
            if self._backtrace and function and function.endswith(self._catch_point_identifier):
                function = function[:-len(self._catch_point_identifier)]
                pattern = '>' + pattern[1:]
            if self._colorize and is_mine:
                dirname, basename = os.path.split(file)
                if dirname:
                    dirname += os.sep
                dirname = self._theme['dirname'].format(dirname)
                basename = self._theme['basename'].format(basename)
                file = dirname + basename
                line = self._theme['line'].format(line)
                function = self._theme['function'].format(function)
            if self._diagnose and (is_mine or prepend_with_new_line):
                pattern = '\n' + pattern
            location = pattern.format(file, line, function)
            frame = location + frame[match.end():]
            prepend_with_new_line = is_mine
        yield frame