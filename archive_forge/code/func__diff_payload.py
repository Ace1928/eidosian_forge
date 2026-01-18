from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def _diff_payload(curr_resp, update_resp, payload_cifs, schedule, job_det):
    diff = 0
    update_resp['ConsoleSetting'].extend(payload_cifs['ConsoleSetting'])
    if schedule and job_det['Schedule'] != schedule:
        diff += 1
    for i in curr_resp:
        for j in update_resp['ConsoleSetting']:
            if i['Name'] == j['Name']:
                if isinstance(j['Value'], bool):
                    j['Value'] = str(j['Value']).lower()
                if isinstance(j['Value'], int):
                    j['Value'] = str(j['Value'])
                if i['Value'] != j['Value']:
                    diff += 1
    return diff