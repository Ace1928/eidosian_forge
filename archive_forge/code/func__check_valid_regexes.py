from __future__ import absolute_import, division, print_function
import re
from ansible.plugins.callback import CallbackBase
from ansible_collections.ansible.utils.plugins.plugin_utils.base.fact_diff import FactDiffBase
def _check_valid_regexes(self):
    if self._skip_lines:
        self._debug("Checking regex in 'split_lines' for validity")
        for idx, regex in enumerate(self._skip_lines):
            try:
                self._skip_lines[idx] = re.compile(regex)
            except re.error as exc:
                msg = "The regex '{regex}', is not valid. The error was {err}.".format(regex=regex, err=str(exc))
                self._errors.append(msg)