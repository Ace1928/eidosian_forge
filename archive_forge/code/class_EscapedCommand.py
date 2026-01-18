import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class EscapedCommand(TokenTransformBase):
    """Transformer for escaped commands like %foo, !foo, or /foo"""

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first escaped command (%foo, !foo, etc.) in the cell.
        """
        for line in tokens_by_line:
            if not line:
                continue
            ix = 0
            ll = len(line)
            while ll > ix and line[ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                ix += 1
            if ix >= ll:
                continue
            if line[ix].string in ESCAPE_SINGLES:
                return cls(line[ix].start)

    def transform(self, lines):
        """Transform an escaped line found by the ``find()`` classmethod.
        """
        start_line, start_col = (self.start_line, self.start_col)
        indent = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        line = assemble_continued_line(lines, (start_line, start_col), end_line)
        if len(line) > 1 and line[:2] in ESCAPE_DOUBLES:
            escape, content = (line[:2], line[2:])
        else:
            escape, content = (line[:1], line[1:])
        if escape in tr:
            call = tr[escape](content)
        else:
            call = ''
        lines_before = lines[:start_line]
        new_line = indent + call + '\n'
        lines_after = lines[end_line + 1:]
        return lines_before + [new_line] + lines_after