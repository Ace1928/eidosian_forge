from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _combine_headers(self):
    """
        This will combine a header default with headers that come from the module declaration
        :return: A combined headers dict
        """
    default_headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
    if self.module.params.get('headers') is not None:
        result = default_headers.copy()
        result.update(self.module.params.get('headers'))
    else:
        result = default_headers
    return result