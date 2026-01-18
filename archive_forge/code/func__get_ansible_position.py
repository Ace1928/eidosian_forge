from __future__ import (absolute_import, division, print_function)
import sys as _sys
from collections.abc import Sequence
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
def _get_ansible_position(self):
    return (self._data_source, self._line_number, self._column_number)