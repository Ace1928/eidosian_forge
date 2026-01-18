from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPSnmpTraphosts:
    """Class with SNMP methods"""

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), host=dict(required=True, type='str', aliases=['ip_address'])))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_snmp_traphosts', 9, 7)

    def get_snmp_traphosts(self):
        query = {'host': self.parameters.get('host'), 'fields': 'host'}
        api = 'support/snmp/traphosts'
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching snmp traphosts info: %s' % error)
        return record

    def create_snmp_traphost(self):
        api = 'support/snmp/traphosts'
        params = {'host': self.parameters.get('host')}
        dummy, error = rest_generic.post_async(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error creating traphost: %s' % error)

    def delete_snmp_traphost(self):
        api = 'support/snmp/traphosts'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters.get('host'))
        if error is not None:
            self.module.fail_json(msg='Error deleting traphost: %s' % error)

    def apply(self):
        """
        Apply action to SNMP traphost
        """
        current = self.get_snmp_traphosts()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_snmp_traphost()
            elif cd_action == 'delete':
                self.delete_snmp_traphost()
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)