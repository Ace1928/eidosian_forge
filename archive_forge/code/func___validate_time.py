from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __validate_time(self, mtime):
    curr_time, date_offset = get_current_time(self.idrac)
    if not mtime.endswith(date_offset):
        self.module.exit_json(failed=True, msg=MAINTENACE_OFFSET_DIFF_MSG.format(date_offset))
    if mtime < curr_time:
        self.module.exit_json(failed=True, msg=MAINTENACE_OFFSET_BEHIND_MSG)