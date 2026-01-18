import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class MagicAssign(TokenTransformBase):
    """Transformer for assignments from magics (a = %foo)"""

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first magic assignment (a = %foo) in the cell.
        """
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if assign_ix is not None and len(line) >= assign_ix + 2 and (line[assign_ix + 1].string == '%') and (line[assign_ix + 2].type == tokenize.NAME):
                return cls(line[assign_ix + 1].start)

    def transform(self, lines: List[str]):
        """Transform a magic assignment found by the ``find()`` classmethod.
        """
        start_line, start_col = (self.start_line, self.start_col)
        lhs = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        rhs = assemble_continued_line(lines, (start_line, start_col), end_line)
        assert rhs.startswith('%'), rhs
        magic_name, _, args = rhs[1:].partition(' ')
        lines_before = lines[:start_line]
        call = 'get_ipython().run_line_magic({!r}, {!r})'.format(magic_name, args)
        new_line = lhs + call + '\n'
        lines_after = lines[end_line + 1:]
        return lines_before + [new_line] + lines_after