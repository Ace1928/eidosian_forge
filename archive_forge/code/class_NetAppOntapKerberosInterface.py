from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapKerberosInterface:
    """Modify Kerberos interface"""

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), interface_name=dict(required=True, type='str'), vserver=dict(required=True, type='str'), enabled=dict(required=True, type='bool'), keytab_uri=dict(required=False, type='str', no_log=True), machine_account=dict(required=False, type='str'), organizational_unit=dict(required=False, type='str'), admin_username=dict(required=False, type='str'), admin_password=dict(required=False, type='str', no_log=True), service_principal_name=dict(required=False, type='str', aliases=['spn'])))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, required_if=[('enabled', True, ['service_principal_name'])], required_together=[('admin_username', 'admin_password')], mutually_exclusive=[('keytab_uri', 'machine_account')])
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_kerberos_interface', 9, 7)
        self.rest_api.is_rest_supported_properties(self.parameters, None, [['machine_account', (9, 12, 1)]])
        self.uuid = None

    def get_kerberos_interface(self):
        """
        Get kerberos interface.
        """
        api = 'protocols/nfs/kerberos/interfaces'
        query = {'interface.name': self.parameters['interface_name'], 'svm.name': self.parameters['vserver'], 'fields': 'interface.uuid,enabled,spn'}
        if 'machine_account' in self.parameters:
            query['fields'] += ',machine_account'
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error fetching kerberos interface info %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())
        if record is None:
            self.module.fail_json(msg='Error: Kerberos interface config does not exist for %s' % self.parameters['interface_name'])
        self.uuid = self.na_helper.safe_get(record, ['interface', 'uuid'])
        return {'enabled': record.get('enabled')}

    def modify_kerberos_interface(self):
        """
        Modify kerberos interface.
        """
        api = 'protocols/nfs/kerberos/interfaces'
        body = {'enabled': self.parameters['enabled']}
        if 'keytab_uri' in self.parameters:
            body['keytab_uri'] = self.parameters['keytab_uri']
        if 'organizational_unit' in self.parameters:
            body['organizational_unit'] = self.parameters['organizational_unit']
        if 'service_principal_name' in self.parameters:
            body['spn'] = self.parameters['service_principal_name']
        if 'admin_username' in self.parameters:
            body['user'] = self.parameters['admin_username']
        if 'admin_password' in self.parameters:
            body['password'] = self.parameters['admin_password']
        if 'machine_account' in self.parameters:
            body['machine_account'] = self.parameters['machine_account']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
        if error:
            self.module.fail_json(msg='Error modifying kerberos interface %s: %s.' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        modify = self.na_helper.get_modified_attributes(self.get_kerberos_interface(), self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_kerberos_interface()
        result = netapp_utils.generate_result(self.na_helper.changed, modify=modify)
        self.module.exit_json(**result)