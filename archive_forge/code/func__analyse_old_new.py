import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
def _analyse_old_new(self):
    if (self._old_lines, self._new_lines) == ([], []):
        return
    if not self.check_style:
        return
    old = [l.contents for l in self._old_lines]
    new = [l.contents for l in self._new_lines]
    ws_matched = self._matched_lines(old, new)
    old = [l.rstrip() for l in old]
    new = [l.rstrip() for l in new]
    no_ws_matched = self._matched_lines(old, new)
    if no_ws_matched < ws_matched:
        raise AssertionError
    if no_ws_matched > ws_matched:
        self.spurious_whitespace += no_ws_matched - ws_matched
        self.target.write('^ Spurious whitespace change above.\n')
    self._old_lines, self._new_lines = ([], [])