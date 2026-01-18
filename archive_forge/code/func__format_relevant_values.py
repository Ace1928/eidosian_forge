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
def _format_relevant_values(self, relevant_values, colorize):
    for i in reversed(range(len(relevant_values))):
        col, value = relevant_values[i]
        pipe_cols = [pcol for pcol, _ in relevant_values[:i]]
        pre_line = ''
        index = 0
        for pc in pipe_cols:
            pre_line += ' ' * (pc - index) + self._pipe_char
            index = pc + 1
        pre_line += ' ' * (col - index)
        value_lines = value.split('\n')
        for n, value_line in enumerate(value_lines):
            if n == 0:
                arrows = pre_line + self._cap_char + ' '
            else:
                arrows = pre_line + ' ' * (len(self._cap_char) + 1)
            if colorize:
                arrows = self._theme['arrows'].format(arrows)
                value_line = self._theme['value'].format(value_line)
            yield (arrows + value_line)