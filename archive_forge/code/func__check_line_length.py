import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _check_line_length(self, part, spacing):
    if part.type == 'backslash':
        last_column = part.start_pos[1] + 1
    else:
        last_column = part.end_pos[1]
    if last_column > self._config.max_characters and spacing.start_pos[1] <= self._config.max_characters:
        report = True
        if part.type == 'comment':
            splitted = part.value[1:].split()
            if len(splitted) == 1 and part.end_pos[1] - len(splitted[0]) < 72:
                report = False
        if report:
            self.add_issue(part, 501, 'Line too long (%s > %s characters)' % (last_column, self._config.max_characters))