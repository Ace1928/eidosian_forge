from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __get_redfish_apply_time(self, aplytm, rf_settings):
    rf_set = {}
    if rf_settings:
        if aplytm not in rf_settings:
            self.module.exit_json(failed=True, msg=APPLY_TIME_NOT_SUPPORTED_MSG.format(aplytm))
        elif 'Maintenance' in aplytm:
            rf_set['ApplyTime'] = aplytm
            m_win = self.module.params.get('maintenance_window')
            self.__validate_time(m_win.get('start_time'))
            rf_set['MaintenanceWindowStartTime'] = m_win.get('start_time')
            rf_set['MaintenanceWindowDurationInSeconds'] = m_win.get('duration')
        else:
            rf_set['ApplyTime'] = aplytm
    return rf_set