from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def _values_is_equal(self, a, b):
    """Expects two string values. It will split the string by whitespace
        and compare each value. It will return True if both lists are the same,
        contain the same elements and the same order."""
    if a is None or b is None:
        return False
    a = a.split()
    b = b.split()
    if len(a) != len(b):
        return False
    return len([i for i, j in zip(a, b) if i == j]) == len(a)