from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgOrgUserS3Key(object):
    """
    Create, modify and delete StorageGRID Tenant Account
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), unique_user_name=dict(required=True, type='str'), expires=dict(required=False, type='str'), access_key=dict(required=False, type='str', no_log=False)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'absent', ['access_key'])], supports_check_mode=False)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        self.data['expires'] = self.parameters.get('expires')

    def get_org_user_id(self, unique_name):
        api = 'api/v3/org/users/%s' % unique_name
        response, error = self.rest_api.get(api)
        if error:
            if response['code'] != 404:
                self.module.fail_json(msg=error)
        else:
            return response['data']['id']
        return None

    def get_org_user_s3_key(self, user_id, access_key):
        api = 'api/v3/org/users/current-user/s3-access-keys/%s' % access_key
        if user_id:
            api = 'api/v3/org/users/%s/s3-access-keys/%s' % (user_id, access_key)
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        else:
            return response['data']
        return None

    def create_org_user_s3_key(self, user_id):
        api = 'api/v3/org/users/current-user/s3-access-keys'
        if user_id:
            api = 'api/v3/org/users/%s/s3-access-keys' % user_id
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_org_user_s3_key(self, user_id, access_key):
        api = 'api/v3/org/users/current-user/s3-access-keys'
        if user_id:
            api = 'api/v3/org/users/%s/s3-access-keys/%s' % (user_id, access_key)
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        result_message = ''
        resp_data = {}
        user_id = None
        if self.parameters.get('unique_user_name'):
            user_id = self.get_org_user_id(self.parameters['unique_user_name'])
        if self.parameters['state'] == 'present':
            org_user_s3_key = None
            if self.parameters.get('access_key'):
                org_user_s3_key = self.get_org_user_s3_key(user_id, self.parameters['access_key'])
                resp_data = org_user_s3_key
            if not org_user_s3_key:
                resp_data = self.create_org_user_s3_key(user_id)
                self.na_helper.changed = True
        if self.parameters['state'] == 'absent':
            self.delete_org_user_s3_key(user_id, self.parameters['access_key'])
            self.na_helper.changed = True
            result_message = 'Org User S3 key deleted'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)