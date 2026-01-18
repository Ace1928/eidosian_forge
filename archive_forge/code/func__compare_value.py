from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
def _compare_value(self, req_value, resp_value):
    diff = None
    if resp_value is None:
        return None
    try:
        if isinstance(req_value, list):
            diff = self._compare_lists(req_value, resp_value)
        elif isinstance(req_value, dict):
            diff = self._compare_dicts(req_value, resp_value)
        elif isinstance(req_value, bool):
            diff = self._compare_boolean(req_value, resp_value)
        elif to_text(req_value) != to_text(resp_value):
            diff = req_value
    except UnicodeError:
        pass
    return diff