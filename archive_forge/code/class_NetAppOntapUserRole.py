from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapUserRole:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), command_directory_name=dict(required=False, type='str'), access_level=dict(required=False, type='str', default='all', choices=['none', 'readonly', 'all']), vserver=dict(required=False, type='str'), query=dict(required=False, type='str'), privileges=dict(required=False, type='list', elements='dict', options=dict(query=dict(required=False, type='str'), access=dict(required=False, type='str', default='all', choices=['none', 'readonly', 'all', 'read_create', 'read_modify', 'read_create_modify']), path=dict(required=True, type='str')))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, mutually_exclusive=[('command_directory_name', 'privileges'), ('access_level', 'privileges'), ('query', 'privileges')])
        self.owner_uuid = None
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if self.parameters.get('privileges') is not None:
            self.parameters['privileges'] = self.na_helper.filter_out_none_entries(self.parameters['privileges'])
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        partially_supported_rest_properties = [['query', (9, 11, 1)], ['privileges.query', (9, 11, 1)], ['command_directory_name', (9, 11, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, None, partially_supported_rest_properties)
        if self.use_rest and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 7, 0)):
            msg = 'REST requires ONTAP 9.7 or later for security/roles APIs.'
            self.use_rest = self.na_helper.fall_back_to_zapi(self.module, msg, self.parameters)
        if not self.use_rest:
            if netapp_utils.has_netapp_lib() is False:
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            if not self.parameters.get('vserver'):
                self.module.fail_json(msg='Error: vserver is required field with ZAPI.')
            if not self.parameters.get('command_directory_name'):
                self.module.fail_json(msg='Error: command_directory_name is required field with ZAPI')
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
        elif not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1) and self.parameters['state'] == 'present':
            self.validate_rest_path()

    def validate_rest_path(self):
        """
        REST does not support command or command directory path in ONTAP < 9.11.1 versions.
        """
        invalid_uri = []
        for privilege in self.parameters.get('privileges', []):
            if '/' not in privilege['path']:
                invalid_uri.append(privilege['path'])
        if invalid_uri:
            self.module.fail_json(msg='Error: Invalid URI %s, please set valid REST API path' % invalid_uri)

    def get_role(self):
        """
        Checks if the role exists for specific command-directory-name.

        :return:
            True if role found
            False if role is not found
        :rtype: bool
        """
        if self.use_rest:
            return self.get_role_rest()
        options = {'vserver': self.parameters['vserver'], 'role-name': self.parameters['name'], 'command-directory-name': self.parameters['command_directory_name']}
        security_login_role_get_iter = netapp_utils.zapi.NaElement('security-login-role-get-iter')
        query_details = netapp_utils.zapi.NaElement.create_node_with_children('security-login-role-info', **options)
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(query_details)
        security_login_role_get_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(security_login_role_get_iter, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as e:
            if to_native(e.code) == '16031':
                return None
            elif to_native(e.code) == '16039':
                return None
            else:
                self.module.fail_json(msg='Error getting role %s: %s' % (self.parameters['name'], to_native(e)), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            role_info = result.get_child_by_name('attributes-list').get_child_by_name('security-login-role-info')
            result = {'name': role_info['role-name'], 'access_level': role_info['access-level'], 'command_directory_name': role_info['command-directory-name'], 'query': role_info['role-query']}
            return result
        return None

    def create_role(self):
        if self.use_rest:
            return self.create_role_rest()
        options = {'vserver': self.parameters['vserver'], 'role-name': self.parameters['name'], 'command-directory-name': self.parameters['command_directory_name'], 'access-level': self.parameters['access_level']}
        if self.parameters.get('query'):
            options['role-query'] = self.parameters['query']
        role_create = netapp_utils.zapi.NaElement.create_node_with_children('security-login-role-create', **options)
        try:
            self.server.invoke_successfully(role_create, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_role(self):
        if self.use_rest:
            return self.delete_role_rest()
        role_delete = netapp_utils.zapi.NaElement.create_node_with_children('security-login-role-delete', **{'vserver': self.parameters['vserver'], 'role-name': self.parameters['name'], 'command-directory-name': self.parameters['command_directory_name']})
        try:
            self.server.invoke_successfully(role_delete, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error removing role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_role(self, modify):
        if self.use_rest:
            return self.modify_role_rest(modify)
        options = {'vserver': self.parameters['vserver'], 'role-name': self.parameters['name'], 'command-directory-name': self.parameters['command_directory_name']}
        if 'access_level' in modify.keys():
            options['access-level'] = self.parameters['access_level']
        if 'query' in modify.keys():
            options['role-query'] = self.parameters['query']
        role_modify = netapp_utils.zapi.NaElement.create_node_with_children('security-login-role-modify', **options)
        try:
            self.server.invoke_successfully(role_modify, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def get_role_rest(self):
        api = 'security/roles'
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1):
            fields = 'name,owner,privileges.path,privileges.access,privileges.query'
        else:
            fields = 'name,owner,privileges.path,privileges.access'
        params = {'name': self.parameters['name'], 'fields': fields}
        if self.parameters.get('vserver'):
            params['owner.name'] = self.parameters['vserver']
        else:
            params['scope'] = 'cluster'
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error getting role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return self.format_record(record)

    def format_record(self, record):
        if not record:
            return None
        for each in self.na_helper.safe_get(record, ['privileges']):
            if each['path'] == 'DEFAULT':
                record['privileges'].remove(each)
        for each in self.na_helper.safe_get(record, ['privileges']):
            if each.get('_links'):
                each.pop('_links')
        return_record = {'name': self.na_helper.safe_get(record, ['name']), 'privileges': self.na_helper.safe_get(record, ['privileges'])}
        self.owner_uuid = self.na_helper.safe_get(record, ['owner', 'uuid'])
        return return_record

    def create_role_rest(self):
        api = 'security/roles'
        body = {'name': self.parameters['name']}
        if self.parameters.get('vserver'):
            body['owner.name'] = self.parameters['vserver']
        body['privileges'] = self.parameters['privileges']
        dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_role_rest(self):
        api = 'security/roles'
        uuids = '%s/%s' % (self.owner_uuid, self.parameters['name'])
        dummy, error = rest_generic.delete_async(self.rest_api, api, uuids, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error deleting role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_role_rest(self, modify):
        privileges = self.get_role_privileges_rest()
        modify_privilege = []
        for privilege in modify['privileges']:
            path = privilege['path']
            modify_privilege.append(path)
            if path not in privileges:
                self.create_role_privilege(privilege)
            elif privilege.get('query'):
                if not privileges[path].get('query'):
                    self.modify_role_privilege(privilege, path)
                elif privilege['query'] != privileges[path]['query']:
                    self.modify_role_privilege(privilege, path)
            elif privilege.get('access') and privilege['access'] != privileges[path]['access']:
                self.modify_role_privilege(privilege, path)
        for privilege_path in privileges:
            if privilege_path not in modify_privilege:
                self.delete_role_privilege(privilege_path)

    def get_role_privileges_rest(self):
        api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api, {})
        if error:
            self.module.fail_json(msg='Error getting role privileges for role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return self.format_privileges(records)

    def format_privileges(self, records):
        return_dict = {}
        for record in records:
            return_dict[record['path']] = record
        return return_dict

    def create_role_privilege(self, privilege):
        api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
        body = {'path': privilege['path'], 'access': privilege['access']}
        dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating role privilege %s: %s' % (privilege['path'], to_native(error)), exception=traceback.format_exc())

    def modify_role_privilege(self, privilege, path):
        path = path.replace('/', '%2F')
        api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
        body = {}
        if privilege.get('access'):
            body['access'] = privilege['access']
        if privilege.get('query'):
            body['query'] = privilege['query']
        dummy, error = rest_generic.patch_async(self.rest_api, api, path, body)
        if error:
            self.module.fail_json(msg='Error modifying privileges for path %s: %s' % (path, to_native(error)), exception=traceback.format_exc())

    def delete_role_privilege(self, path):
        path = path.replace('/', '%2F')
        api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
        dummy, error = rest_generic.delete_async(self.rest_api, api, path, job_timeout=120)
        if error:
            if "entry doesn't exist" in error and "'target': 'path'" in error:
                return
            self.module.fail_json(msg='Error deleting role privileges %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def convert_parameters(self):
        if self.parameters.get('privileges') is not None:
            return
        if not self.parameters.get('command_directory_name'):
            self.module.fail_json(msg='Error: either path or command_directory_name is required in REST.')
        self.parameters['privileges'] = []
        temp_dict = {'path': self.parameters['command_directory_name'], 'access': self.parameters['access_level']}
        self.parameters.pop('command_directory_name')
        self.parameters.pop('access_level')
        if self.parameters.get('query'):
            temp_dict['query'] = self.parameters['query']
            self.parameters.pop('query')
        self.parameters['privileges'] = [temp_dict]

    def validate_create_modify_required(self, current, modify):
        new_current = self.get_role()
        new_cd_action = self.na_helper.get_cd_action(new_current, self.parameters)
        new_modify = None if new_cd_action else self.na_helper.get_modified_attributes(new_current, self.parameters)
        msg = ''
        if current is None and new_modify:
            msg = 'Create operation also affected additional related commands: %s' % new_current['privileges']
        elif modify and new_cd_action == 'create':
            msg = "Create role is required, desired is: %s but it's a subset of relevant commands/command directory configured in current: %s,\n                     deleting one of the commands will remove all the commands in the relevant group" % (self.parameters['privileges'], current['privileges'])
        elif modify and new_modify:
            msg = 'modify is required, desired: %s and new current: %s' % (self.parameters['privileges'], new_current['privileges'])
        if msg:
            self.module.warn(msg)

    def apply(self):
        if self.use_rest:
            self.convert_parameters()
        current = self.get_role()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if not self.use_rest and self.parameters.get('query') == '' and (current is not None) and (current['query'] is None):
            self.parameters['query'] = None
        modify = None if cd_action else self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_role()
            elif cd_action == 'delete':
                self.delete_role()
            elif modify:
                self.modify_role(modify)
            if self.use_rest:
                self.validate_create_modify_required(current, modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)