from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
class OEMNetworkAttributes(IDRACNetworkAttributes):

    def __init__(self, idrac, module):
        super().__init__(idrac, module)

    def clear_pending(self):
        firm_ver = get_idrac_firmware_version(self.idrac)
        oem_network_attributes = self.module.params.get('oem_network_attributes')
        if LooseVersion(firm_ver) < '3.0':
            if oem_network_attributes:
                return None
            self.module.exit_json(msg=CLEAR_PENDING_NOT_SUPPORTED_WITHOUT_ATTR_IDRAC8)
        resp = get_dynamic_uri(self.idrac, self.oem_uri, '@Redfish.Settings')
        settings_uri = resp.get('SettingsObject').get('@odata.id')
        settings_uri_resp = get_dynamic_uri(self.idrac, settings_uri)
        pending_attributes = settings_uri_resp.get('Attributes')
        clear_pending_uri = settings_uri_resp.get('Actions').get('#DellManager.ClearPending').get('target')
        if not pending_attributes and (not oem_network_attributes):
            self.module.exit_json(msg=NO_CHANGES_FOUND_MSG)
        job_resp = get_scheduled_job_resp(self.idrac, 'NICConfiguration')
        job_id, job_state = (job_resp.get('Id'), job_resp.get('JobState'))
        if job_id:
            if job_state in ['Running']:
                job_resp = remove_key(job_resp, regex_pattern='(.*?)@odata')
                self.module.exit_json(failed=True, msg=JOB_RUNNING_CLEAR_PENDING_ATTR.format('NICConfiguration'), job_status=job_resp)
            elif job_state in ['Starting', 'Scheduled', 'Scheduling']:
                if self.module.check_mode and (not oem_network_attributes):
                    self.module.exit_json(msg=CHANGES_FOUND_MSG, changed=True)
                if not self.module.check_mode:
                    delete_job(self.idrac, job_id)
        if self.module.check_mode and (not oem_network_attributes):
            self.module.exit_json(msg=CHANGES_FOUND_MSG, changed=True)
        time.sleep(5)
        settings_uri_resp = get_dynamic_uri(self.idrac, settings_uri)
        pending_attributes = settings_uri_resp.get('Attributes')
        if pending_attributes and (not self.module.check_mode):
            self.idrac.invoke_request(clear_pending_uri, 'POST', data='{}', dump=False)
        if not oem_network_attributes:
            self.module.exit_json(msg=SUCCESS_CLEAR_PENDING_ATTR_MSG, changed=True)

    def perform_operation(self):
        oem_network_attributes = self.module.params.get('oem_network_attributes')
        network_device_function_id = self.module.params.get('network_device_function_id')
        apply_time = self.module.params.get('apply_time')
        job_wait = self.module.params.get('job_wait')
        invalid_attr = {}
        firm_ver = get_idrac_firmware_version(self.idrac)
        if LooseVersion(firm_ver) < '3.0':
            root = '<SystemConfiguration>{0}</SystemConfiguration>'
            scp_payload = root.format(xml_data_conversion(oem_network_attributes, network_device_function_id))
            resp = self.idrac.import_scp(import_buffer=scp_payload, target='NIC', job_wait=False)
        else:
            payload = {'Attributes': oem_network_attributes}
            apply_time_setting = self.apply_time(self.oem_uri)
            if apply_time_setting:
                payload.update({'@Redfish.SettingsApplyTime': apply_time_setting})
            patch_uri = get_dynamic_uri(self.idrac, self.oem_uri).get('@Redfish.Settings').get('SettingsObject').get('@odata.id')
            resp = self.idrac.invoke_request(method='PATCH', uri=patch_uri, data=payload)
            job_wait = job_wait if apply_time == 'Immediate' else False
        invalid_attr = self.extract_error_msg(resp)
        return (resp, invalid_attr, job_wait)