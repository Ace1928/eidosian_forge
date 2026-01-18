from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
class NetAppOntapCifsLocalUser:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), vserver=dict(required=True, type='str'), account_disabled=dict(required=False, type='bool'), full_name=dict(required=False, type='str'), description=dict(required=False, type='str'), user_password=dict(required=False, type='str', no_log=True), set_password=dict(required=False, type='bool', default=False)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.svm_uuid = None
        self.sid = None
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_cifs_local_user', 9, 10, 1)

    def get_cifs_local_user(self):
        self.get_svm_uuid()
        api = 'protocols/cifs/local-users'
        fields = 'account_disabled,description,full_name,name,sid'
        params = {'svm.uuid': self.svm_uuid, 'name': self.parameters['name'], 'fields': fields}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching cifs/local-user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if record:
            return self.format_record(record)
        return None

    def get_svm_uuid(self):
        self.svm_uuid, dummy = rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)

    def format_record(self, record):
        self.sid = record['sid']
        try:
            record['name'] = record['name'].split('\\')[1]
        except SyntaxError:
            self.module.fail_json(msg='Error fetching cifs/local-user')
        return record

    def create_cifs_local_user(self):
        api = 'protocols/cifs/local-users'
        body = {'svm.uuid': self.svm_uuid, 'name': self.parameters['name']}
        if self.parameters.get('user_password') is not None:
            body['password'] = self.parameters['user_password']
        if self.parameters.get('full_name') is not None:
            body['full_name'] = self.parameters['full_name']
        if self.parameters.get('description') is not None:
            body['description'] = self.parameters['description']
        if self.parameters.get('account_disabled') is not None:
            body['account_disabled'] = self.parameters['account_disabled']
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating CIFS local users with name %s: %s' % (self.parameters['name'], error))

    def delete_cifs_local_user(self):
        api = 'protocols/cifs/local-users'
        uuids = '%s/%s' % (self.svm_uuid, self.sid)
        dummy, error = rest_generic.delete_async(self.rest_api, api, uuids)
        if error:
            self.module.fail_json(msg='Error while deleting CIFS local user: %s' % error)

    def modify_cifs_local_user(self, modify):
        api = 'protocols/cifs/local-users'
        uuids = '%s/%s' % (self.svm_uuid, self.sid)
        body = {}
        if modify.get('full_name') is not None:
            body['full_name'] = self.parameters['full_name']
        if modify.get('description') is not None:
            body['description'] = self.parameters['description']
        if modify.get('account_disabled') is not None:
            body['account_disabled'] = self.parameters['account_disabled']
        if self.parameters['set_password'] and modify.get('user_password') is not None:
            body['password'] = self.parameters['user_password']
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuids, body)
        if error:
            self.module.fail_json(msg='Error while modifying CIFS local user: %s' % error)

    def apply(self):
        current = self.get_cifs_local_user()
        modify = None
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if self.parameters['set_password'] and self.parameters.get('user_password') is not None:
                if not modify:
                    modify = {}
                    self.na_helper.changed = True
                modify.update({'user_password': self.parameters['user_password']})
                self.module.warn('forcing a password change as set_password is true')
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_cifs_local_user()
            elif cd_action == 'delete':
                self.delete_cifs_local_user()
            elif modify:
                self.modify_cifs_local_user(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)