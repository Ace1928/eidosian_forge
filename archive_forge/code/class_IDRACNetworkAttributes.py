from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
class IDRACNetworkAttributes:

    def __init__(self, idrac, module):
        self.module = module
        self.idrac = idrac
        self.redfish_uri = None
        self.oem_uri = None

    def __perform_validation_for_network_adapter_id(self):
        odata = '@odata.id'
        network_adapter_id = self.module.params.get('network_adapter_id')
        network_adapter_id_uri, found_adapter = ('', False)
        uri, error_msg = validate_and_get_first_resource_id_uri(self.module, self.idrac, SYSTEMS_URI)
        if error_msg:
            self.module.exit_json(msg=error_msg, failed=True)
        network_adapters = get_dynamic_uri(self.idrac, uri, 'NetworkInterfaces')[odata]
        network_adapter_list = get_dynamic_uri(self.idrac, network_adapters, 'Members')
        for each_adapter in network_adapter_list:
            if network_adapter_id in each_adapter.get(odata):
                found_adapter = True
                network_adapter_id_uri = each_adapter.get(odata)
                break
        if not found_adapter:
            self.module.exit_json(failed=True, msg=INVALID_ID_MSG.format(network_adapter_id, 'network_adapter_id'))
        return network_adapter_id_uri

    def __perform_validation_for_network_device_function_id(self):
        odata = '@odata.id'
        network_device_function_id_uri, found_device = ('', False)
        network_device_function_id = self.module.params.get('network_device_function_id')
        network_adapter_id_uri = self.__perform_validation_for_network_adapter_id()
        network_devices = get_dynamic_uri(self.idrac, network_adapter_id_uri, 'NetworkDeviceFunctions')[odata]
        network_device_list = get_dynamic_uri(self.idrac, network_devices, 'Members')
        for each_device in network_device_list:
            if network_device_function_id in each_device.get(odata):
                found_device = True
                network_device_function_id_uri = each_device.get(odata)
                break
        if not found_device:
            self.module.exit_json(failed=True, msg=INVALID_ID_MSG.format(network_device_function_id, 'network_device_function_id'))
        return network_device_function_id_uri

    def __get_registry_fw_less_than_6_more_than_3(self):
        reg = {}
        network_device_function_id = self.module.params.get('network_device_function_id')
        registry = get_dynamic_uri(self.idrac, REGISTRY_URI, 'Members')
        for each_member in registry:
            if network_device_function_id in each_member.get('@odata.id'):
                location = get_dynamic_uri(self.idrac, each_member.get('@odata.id'), 'Location')
                if location:
                    uri = location[0].get('Uri')
                    attr = get_dynamic_uri(self.idrac, uri, 'RegistryEntries').get('Attributes', {})
                    for each_attr in attr:
                        reg.update({each_attr['AttributeName']: each_attr['CurrentValue']})
                    break
        return reg

    def __validate_time(self, mtime):
        curr_time, date_offset = get_current_time(self.idrac)
        if not mtime.endswith(date_offset):
            self.module.exit_json(failed=True, msg=MAINTENACE_OFFSET_DIFF_MSG.format(date_offset))
        if mtime < curr_time:
            self.module.exit_json(failed=True, msg=MAINTENACE_OFFSET_BEHIND_MSG)

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

    def __get_registry_fw_less_than_3(self):
        reg = {}
        network_device_function_id = self.module.params.get('network_device_function_id')
        scp_response = self.idrac.export_scp(export_format='JSON', export_use='Default', target='NIC', job_wait=True)
        comp = scp_response.json_data.get('SystemConfiguration', {}).get('Components', {})
        for each in comp:
            if each.get('FQDD') == network_device_function_id:
                for each_attr in each.get('Attributes'):
                    reg.update({each_attr['Name']: each_attr['Value']})
        return reg

    def get_current_server_registry(self):
        reg = {}
        oem_network_attributes = self.module.params.get('oem_network_attributes')
        network_attributes = self.module.params.get('network_attributes')
        firm_ver = get_idrac_firmware_version(self.idrac)
        if oem_network_attributes:
            if LooseVersion(firm_ver) >= '6.0':
                reg = get_dynamic_uri(self.idrac, self.oem_uri, 'Attributes')
            elif '3.0' < LooseVersion(firm_ver) < '6.0':
                reg = self.__get_registry_fw_less_than_6_more_than_3()
            else:
                reg = self.__get_registry_fw_less_than_3()
        if network_attributes:
            resp = get_dynamic_uri(self.idrac, self.redfish_uri)
            reg.update({'Ethernet': resp.get('Ethernet', {})})
            reg.update({'FibreChannel': resp.get('FibreChannel', {})})
            reg.update({'iSCSIBoot': resp.get('iSCSIBoot', {})})
        return reg

    def extract_error_msg(self, resp):
        error_info = {}
        if resp.body:
            error = resp.json_data.get('error')
            for each_dict_err in error.get('@Message.ExtendedInfo'):
                key = each_dict_err.get('MessageArgs')[0]
                msg = each_dict_err.get('Message')
                if key not in error_info:
                    error_info.update({key: msg})
        return error_info

    def get_diff_between_current_and_module_input(self, module_attr, server_attr):
        diff, invalid = (0, {})
        if module_attr is None:
            module_attr = {}
        for each_attr in module_attr:
            if each_attr in server_attr:
                data_type = type(server_attr[each_attr])
                if not isinstance(module_attr[each_attr], data_type):
                    diff += 1
                elif isinstance(module_attr[each_attr], dict) and isinstance(server_attr[each_attr], dict):
                    tmp_diff, tmp_invalid = self.get_diff_between_current_and_module_input(module_attr[each_attr], server_attr[each_attr])
                    diff += tmp_diff
                    invalid.update(tmp_invalid)
                elif module_attr[each_attr] != server_attr[each_attr]:
                    diff += 1
            elif each_attr not in server_attr:
                invalid.update({each_attr: ATTRIBUTE_NOT_EXIST_CHECK_IDEMPOTENCY_MODE})
        return (diff, invalid)

    def validate_job_timeout(self):
        if self.module.params.get('job_wait') and self.module.params.get('job_wait_timeout') <= 0:
            self.module.exit_json(msg=TIMEOUT_NEGATIVE_OR_ZERO_MSG, failed=True)

    def apply_time(self, setting_uri):
        resp = get_dynamic_uri(self.idrac, setting_uri, '@Redfish.Settings')
        rf_settings = resp.get('SupportedApplyTimes', [])
        apply_time = self.module.params.get('apply_time', {})
        rf_set = self.__get_redfish_apply_time(apply_time, rf_settings)
        return rf_set

    def set_dynamic_base_uri_and_validate_ids(self):
        network_device_function_id_uri = self.__perform_validation_for_network_device_function_id()
        resp = get_dynamic_uri(self.idrac, network_device_function_id_uri)
        self.oem_uri = resp.get('Links', {}).get('Oem', {}).get('Dell', {}).get('DellNetworkAttributes', {}).get('@odata.id', {})
        self.redfish_uri = network_device_function_id_uri