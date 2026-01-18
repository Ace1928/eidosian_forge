import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _dedentlines(lines, tabsize=8, skip_first_line=False):
    """_dedentlines(lines, tabsize=8, skip_first_line=False) -> dedented lines

        "lines" is a list of lines to dedent.
        "tabsize" is the tab width to use for indent width calculations.
        "skip_first_line" is a boolean indicating if the first line should
            be skipped for calculating the indent width and for dedenting.
            This is sometimes useful for docstrings and similar.

    Same as dedent() except operates on a sequence of lines. Note: the
    lines list is modified **in-place**.
    """
    DEBUG = False
    if DEBUG:
        print('dedent: dedent(..., tabsize=%d, skip_first_line=%r)' % (tabsize, skip_first_line))
    margin = None
    for i, line in enumerate(lines):
        if i == 0 and skip_first_line:
            continue
        indent = 0
        for ch in line:
            if ch == ' ':
                indent += 1
            elif ch == '\t':
                indent += tabsize - indent % tabsize
            elif ch in '\r\n':
                continue
            else:
                break
        else:
            continue
        if DEBUG:
            print('dedent: indent=%d: %r' % (indent, line))
        if margin is None:
            margin = indent
        else:
            margin = min(margin, indent)
    if DEBUG:
        print('dedent: margin=%r' % margin)
    if margin is not None and margin > 0:
        for i, line in enumerate(lines):
            if i == 0 and skip_first_line:
                continue
            removed = 0
            for j, ch in enumerate(line):
                if ch == ' ':
                    removed += 1
                elif ch == '\t':
                    removed += tabsize - removed % tabsize
                elif ch in '\r\n':
                    if DEBUG:
                        print('dedent: %r: EOL -> strip up to EOL' % line)
                    lines[i] = lines[i][j:]
                    break
                else:
                    raise ValueError('unexpected non-whitespace char %r in line %r while removing %d-space margin' % (ch, line, margin))
                if DEBUG:
                    print('dedent: %r: %r -> removed %d/%d' % (line, ch, removed, margin))
                if removed == margin:
                    lines[i] = lines[i][j + 1:]
                    break
                elif removed > margin:
                    lines[i] = ' ' * (removed - margin) + lines[i][j + 1:]
                    break
            else:
                if removed:
                    lines[i] = lines[i][removed:]
    return lines