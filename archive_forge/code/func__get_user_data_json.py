from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.facts import ansible_collector, default_collectors
def _get_user_data_json(self):
    try:
        return yaml.safe_load(self._fetch(CS_USERDATA_BASE_URL))
    except Exception:
        return None