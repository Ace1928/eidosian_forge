from __future__ import absolute_import, division, print_function
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.basic import AnsibleModule
class VmwareVcenterSettingsInfo(PyVmomi):

    def __init__(self, module):
        super(VmwareVcenterSettingsInfo, self).__init__(module)
        self.schema = self.params['schema']
        self.properties = self.params['properties']
        if not self.is_vcenter():
            self.module.fail_json(msg='You have to connect to a vCenter server!')

    def ensure(self):
        result = {}
        exists_vcenter_config = {}
        option_manager = self.content.setting
        for setting in option_manager.setting:
            exists_vcenter_config[setting.key] = setting.value
        if self.schema == 'summary':
            common_name_value_map = {'VirtualCenter.MaxDBConnection': 'db_max_connections_previous', 'task.maxAgeEnabled': 'db_task_cleanup_previous', 'task.maxAge': 'db_task_retention_previous', 'event.maxAgeEnabled': 'db_event_cleanup_previous', 'event.maxAge': 'db_event_retention_previous', 'instance.id': 'runtime_unique_id_previous', 'VirtualCenter.ManagedIP': 'runtime_managed_address_previous', 'VirtualCenter.InstanceName': 'runtime_server_name_previous', 'ads.timeout': 'directory_timeout_previous', 'ads.maxFetchEnabled': 'directory_query_limit_previous', 'ads.maxFetch': 'directory_query_limit_size_previous', 'ads.checkIntervalEnabled': 'directory_validation_previous', 'ads.checkInterval': 'directory_validation_period_previous', 'mail.smtp.server': 'mail_server_previous', 'mail.sender': 'mail_sender_previous', 'snmp.receiver.1.enabled': 'snmp_1_enabled_previous', 'snmp.receiver.1.name': 'snmp_1_url_previous', 'snmp.receiver.1.port': 'snmp_receiver_1_port_previous', 'snmp.receiver.1.community': 'snmp_1_community_previous', 'snmp.receiver.2.enabled': 'snmp_2_enabled_previous', 'snmp.receiver.2.name': 'snmp_2_url_previous', 'snmp.receiver.2.port': 'snmp_receiver_2_port_previous', 'snmp.receiver.2.community': 'snmp_2_community_previous', 'snmp.receiver.3.enabled': 'snmp_3_enabled_previous', 'snmp.receiver.3.name': 'snmp_3_url_previous', 'snmp.receiver.3.port': 'snmp_receiver_3_port_previous', 'snmp.receiver.3.community': 'snmp_3_community_previous', 'snmp.receiver.4.enabled': 'snmp_4_enabled_previous', 'snmp.receiver.4.name': 'snmp_4_url_previous', 'snmp.receiver.4.port': 'snmp_receiver_4_port_previous', 'snmp.receiver.4.community': 'snmp_4_community_previous', 'client.timeout.normal': 'timeout_normal_operations_previous', 'client.timeout.long': 'timeout_long_operations_previous', 'log.level': 'logging_options_previous'}
            for key, value in common_name_value_map.items():
                if key in exists_vcenter_config:
                    result[value] = setting.value
        elif self.properties:
            for property in self.properties:
                if property in exists_vcenter_config:
                    result[property] = exists_vcenter_config[property]
                else:
                    self.module.fail_json(msg="Propety '%s' not found" % property)
        else:
            for property in exists_vcenter_config.keys():
                result[property] = exists_vcenter_config[property]
        self.module.exit_json(changed=False, vcenter_config_info=result)