import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class SystemAssign(TokenTransformBase):
    """Transformer for assignments from system commands (a = !foo)"""

    @classmethod
    def find_pre_312(cls, tokens_by_line):
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if assign_ix is not None and (not line[assign_ix].line.strip().startswith('=')) and (len(line) >= assign_ix + 2) and (line[assign_ix + 1].type == tokenize.ERRORTOKEN):
                ix = assign_ix + 1
                while ix < len(line) and line[ix].type == tokenize.ERRORTOKEN:
                    if line[ix].string == '!':
                        return cls(line[ix].start)
                    elif not line[ix].string.isspace():
                        break
                    ix += 1

    @classmethod
    def find_post_312(cls, tokens_by_line):
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if assign_ix is not None and (not line[assign_ix].line.strip().startswith('=')) and (len(line) >= assign_ix + 2) and (line[assign_ix + 1].type == tokenize.OP) and (line[assign_ix + 1].string == '!'):
                return cls(line[assign_ix + 1].start)

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first system assignment (a = !foo) in the cell."""
        if sys.version_info < (3, 12):
            return cls.find_pre_312(tokens_by_line)
        return cls.find_post_312(tokens_by_line)

    def transform(self, lines: List[str]):
        """Transform a system assignment found by the ``find()`` classmethod.
        """
        start_line, start_col = (self.start_line, self.start_col)
        lhs = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        rhs = assemble_continued_line(lines, (start_line, start_col), end_line)
        assert rhs.startswith('!'), rhs
        cmd = rhs[1:]
        lines_before = lines[:start_line]
        call = 'get_ipython().getoutput({!r})'.format(cmd)
        new_line = lhs + call + '\n'
        lines_after = lines[end_line + 1:]
        return lines_before + [new_line] + lines_after