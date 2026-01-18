from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
class NetworkAttributes(IDRACNetworkAttributes):

    def __init__(self, idrac, module):
        super().__init__(idrac, module)

    def perform_operation(self):
        updatable_fields = ['Ethernet', 'iSCSIBoot', 'FibreChannel']
        network_attributes = self.module.params.get('network_attributes')
        apply_time = self.module.params.get('apply_time')
        job_wait = self.module.params.get('job_wait')
        payload, invalid_attr = ({}, {})
        for each_attr in network_attributes:
            if each_attr in updatable_fields:
                payload.update({each_attr: network_attributes[each_attr]})
        apply_time_setting = self.apply_time(self.redfish_uri)
        if apply_time_setting:
            payload.update({'@Redfish.SettingsApplyTime': apply_time_setting})
        resp = get_dynamic_uri(self.idrac, self.redfish_uri)
        patch_uri = resp.get('@Redfish.Settings', {}).get('SettingsObject', {}).get('@odata.id', {})
        resp = self.idrac.invoke_request(method='PATCH', uri=patch_uri, data=payload)
        invalid_attr = self.extract_error_msg(resp)
        job_wait = job_wait if apply_time == 'Immediate' else False
        return (resp, invalid_attr, job_wait)