import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
def _writeline(self, line):
    item = self.lp.parse_line(line)
    bad_ws_match = None
    if isinstance(item, Hunk):
        line_class = 'diffstuff'
        self._analyse_old_new()
    elif isinstance(item, HunkLine):
        bad_ws_match = re.match(b'^([\\t]*)(.*?)([\\t ]*)(\\r?\\n)$', item.contents)
        has_leading_tabs = bool(bad_ws_match.group(1))
        has_trailing_whitespace = bool(bad_ws_match.group(3))
        if isinstance(item, InsertLine):
            if has_leading_tabs:
                self.added_leading_tabs += 1
            if has_trailing_whitespace:
                self.added_trailing_whitespace += 1
            if len(bad_ws_match.group(2)) > self.max_line_len and (not item.contents.startswith(b'++ ')):
                self.long_lines += 1
            line_class = 'newtext'
            self._new_lines.append(item)
        elif isinstance(item, RemoveLine):
            line_class = 'oldtext'
            self._old_lines.append(item)
        else:
            line_class = 'plain'
    elif isinstance(item, bytes) and item.startswith(b'==='):
        line_class = 'metaline'
        self._analyse_old_new()
    else:
        line_class = 'plain'
        self._analyse_old_new()
    self.colorstring(line_class, item, bad_ws_match)