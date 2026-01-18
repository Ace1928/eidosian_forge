from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def _tuplify_list(element):
    if isinstance(element, list):
        return tuple(element)
    return element