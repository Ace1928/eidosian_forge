from __future__ import absolute_import, division, print_function
import re
from ansible.plugins.callback import CallbackBase
from ansible_collections.ansible.utils.plugins.plugin_utils.base.fact_diff import FactDiffBase
def _xform(self):
    if self._skip_lines:
        if isinstance(self._before, str):
            self._debug("'before' is a string, splitting lines")
            self._before = self._before.splitlines()
        if isinstance(self._after, str):
            self._debug("'after' is a string, splitting lines")
            self._after = self._after.splitlines()
        self._before = [line for line in self._before if not any((regex.match(str(line)) for regex in self._skip_lines))]
        self._after = [line for line in self._after if not any((regex.match(str(line)) for regex in self._skip_lines))]
    if isinstance(self._before, list):
        self._debug("'before' is a list, joining with \n")
        self._before = '\n'.join(map(str, self._before)) + '\n'
    if isinstance(self._after, list):
        self._debug("'after' is a list, joining with \n")
        self._after = '\n'.join(map(str, self._after)) + '\n'