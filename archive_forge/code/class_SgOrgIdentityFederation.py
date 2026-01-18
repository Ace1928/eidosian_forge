from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgOrgIdentityFederation:
    """
    Configure and modify StorageGRID Tenant Identity Federation
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), username=dict(required=False, type='str'), password=dict(required=False, type='str', no_log=True), hostname=dict(required=False, type='str'), port=dict(required=False, type='int'), base_group_dn=dict(required=False, type='str'), base_user_dn=dict(required=False, type='str'), ldap_service_type=dict(required=False, type='str', choices=['OpenLDAP', 'Active Directory', 'Other']), type=dict(required=False, type='str', default='ldap'), ldap_user_id_attribute=dict(required=False, type='str'), ldap_user_uuid_attribute=dict(required=False, type='str'), ldap_group_id_attribute=dict(required=False, type='str'), ldap_group_uuid_attribute=dict(required=False, type='str'), tls=dict(required=False, type='str', choices=['STARTTLS', 'LDAPS', 'Disabled'], default='STARTTLS'), ca_cert=dict(required=False, type='str')))
        parameter_map = {'username': 'username', 'password': 'password', 'hostname': 'hostname', 'port': 'port', 'base_group_dn': 'baseGroupDn', 'base_user_dn': 'baseUserDn', 'ldap_service_type': 'ldapServiceType', 'ldap_user_id_attribute': 'ldapUserIdAttribute', 'ldap_user_uuid_attribute': 'ldapUserUUIDAttribute', 'ldap_group_id_attribute': 'ldapGroupIdAttribute', 'ldap_group_uuid_attribute': 'ldapGroupUUIDAttribute', 'ca_cert': 'caCert'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        if self.parameters['state'] == 'present':
            self.data['disable'] = False
        for k in parameter_map.keys():
            if self.parameters.get(k) is not None:
                self.data[parameter_map[k]] = self.parameters[k]
        if self.parameters.get('tls') == 'STARTTLS':
            self.data['disableTLS'] = False
            self.data['enableLDAPS'] = False
        elif self.parameters.get('tls') == 'LDAPS':
            self.data['disableTLS'] = False
            self.data['enableLDAPS'] = True
        else:
            self.data['disableTLS'] = True
            self.data['enableLDAPS'] = False

    def get_org_identity_source(self):
        api = 'api/v3/org/identity-source'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        else:
            return response['data']
        return None

    def update_identity_federation(self, test=False):
        api = 'api/v3/org/identity-source'
        params = {}
        if test:
            params['test'] = True
        response, error = self.rest_api.put(api, self.data, params=params)
        if error:
            self.module.fail_json(msg=error, payload=self.data)
        if response is not None:
            return response['data']
        else:
            return None

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        org_identity_source = self.get_org_identity_source()
        cd_action = self.na_helper.get_cd_action(org_identity_source, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            for k in (i for i in self.data.keys() if i != 'password'):
                if self.data[k] != org_identity_source.get(k):
                    update = True
                    break
            if self.data.get('password') and self.parameters['state'] == 'present':
                update = True
                self.module.warn('Password attribute has been specified. Task is not idempotent.')
            if update:
                self.na_helper.changed = True
        if cd_action == 'delete':
            if org_identity_source.get('disable'):
                self.na_helper.changed = False
        result_message = ''
        resp_data = org_identity_source
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'delete':
                self.data = dict(disable=True)
                resp_data = self.update_identity_federation()
                result_message = 'Tenant identity federation disabled'
            else:
                resp_data = self.update_identity_federation()
                result_message = 'Tenant identity federation updated'
        if self.module.check_mode:
            self.update_identity_federation(test=True)
            self.module.exit_json(changed=self.na_helper.changed, msg='Connection test successful')
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)