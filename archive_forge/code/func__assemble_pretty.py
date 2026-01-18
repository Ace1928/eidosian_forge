import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
@staticmethod
def _assemble_pretty(first_lines, op, second_lines):
    max_lines = max(len(first_lines), len(second_lines))
    first_lines = _pad_vertically(first_lines, max_lines)
    second_lines = _pad_vertically(second_lines, max_lines)
    blank = ' ' * len(op)
    first_second_lines = list(zip(first_lines, second_lines))
    return [' ' + first_line + ' ' + blank + ' ' + second_line + ' ' for first_line, second_line in first_second_lines[:2]] + ['(' + first_line + ' ' + op + ' ' + second_line + ')' for first_line, second_line in first_second_lines[2:3]] + [' ' + first_line + ' ' + blank + ' ' + second_line + ' ' for first_line, second_line in first_second_lines[3:]]