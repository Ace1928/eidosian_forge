from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _is_object_changed(self, keys, module, result):
    """
        Check if my object is changed
        :param keys: The keys that will determine if an object is changed
        :param module: The module
        :param result: The result from the query
        :return:
        """
    for key in keys:
        if module.params.get(key) != result[key]:
            return True
    return False