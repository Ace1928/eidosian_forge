import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
class LineParser:

    def parse_line(self, line):
        if line.startswith(b'@'):
            return hunk_from_header(line)
        elif line.startswith(b'+'):
            return InsertLine(line[1:])
        elif line.startswith(b'-'):
            return RemoveLine(line[1:])
        elif line.startswith(b' '):
            return ContextLine(line[1:])
        else:
            return line