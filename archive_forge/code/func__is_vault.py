from __future__ import (absolute_import, division, print_function)
import json
import datetime
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.collections_abc import Mapping
from ansible.module_utils.common.collections import is_sequence
def _is_vault(value):
    return getattr(value, '__ENCRYPTED__', False)