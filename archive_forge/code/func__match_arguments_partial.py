import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _match_arguments_partial(self, actions, arg_strings_pattern):
    result = []
    for i in range(len(actions), 0, -1):
        actions_slice = actions[:i]
        pattern = ''.join([self._get_nargs_pattern(action) for action in actions_slice])
        match = _re.match(pattern, arg_strings_pattern)
        if match is not None:
            result.extend([len(string) for string in match.groups()])
            break
    return result