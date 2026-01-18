from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
class _DictComparison(object):
    """ This class takes in two dictionaries `a` and `b`.
        These are dictionaries of arbitrary depth, but made up of standard
        Python types only.
        This differ will compare all values in `a` to those in `b`.
        If value in `a` is None, always returns True, indicating
        this value is no need to compare.
        Note: On all lists, order does matter.
    """

    def __init__(self, request):
        self.request = request

    def __eq__(self, other):
        return self._compare_dicts(self.request, other.request)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _compare_dicts(self, dict1, dict2):
        if dict1 is None:
            return True
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        for k in dict1:
            if not self._compare_value(dict1.get(k), dict2.get(k)):
                return False
        return True

    def _compare_lists(self, list1, list2):
        """Takes in two lists and compares them."""
        if list1 is None:
            return True
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if not self._compare_value(list1[i], list2[i]):
                return False
        return True

    def _compare_value(self, value1, value2):
        """
        return: True: value1 is same as value2, otherwise False.
        """
        if value1 is None:
            return True
        if not (value1 and value2):
            return not value1 and (not value2)
        if isinstance(value1, list) and isinstance(value2, list):
            return self._compare_lists(value1, value2)
        elif isinstance(value1, dict) and isinstance(value2, dict):
            return self._compare_dicts(value1, value2)
        return to_text(value1, errors='surrogate_or_strict') == to_text(value2, errors='surrogate_or_strict')