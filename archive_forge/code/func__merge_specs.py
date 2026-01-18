from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _merge_specs(self, default_specs, custom_specs):
    result = default_specs.copy()
    result.update(custom_specs)
    return result