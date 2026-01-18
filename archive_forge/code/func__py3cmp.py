from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def _py3cmp(a, b):
    """Python 2 can sort lists of mixed types. Strings < tuples. Without this function this fails on Python 3."""
    try:
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0
    except TypeError as e:
        str_ind = to_text(e).find('str')
        tup_ind = to_text(e).find('tuple')
        if -1 not in (str_ind, tup_ind):
            if str_ind < tup_ind:
                return -1
            elif tup_ind < str_ind:
                return 1
        raise