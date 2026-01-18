from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def _transform_dict_keys(self, fact_dict):
    """update a dicts keys to use new names as transformed by self._transform_name"""
    for old_key in list(fact_dict.keys()):
        new_key = self._transform_name(old_key)
        fact_dict[new_key] = fact_dict.pop(old_key)
    return fact_dict