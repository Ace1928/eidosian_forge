from __future__ import (absolute_import, division, print_function)
import difflib
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.common.text.converters import to_text
def _indent_text(self, text, indent_level):
    lines = text.splitlines()
    result_lines = []
    for l in lines:
        result_lines.append('{0}{1}'.format(' ' * indent_level, l))
    return '\n'.join(result_lines)