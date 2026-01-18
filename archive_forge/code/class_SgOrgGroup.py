from __future__ import absolute_import, division, print_function
import json
import re
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgOrgGroup(object):
    """
    Create, modify and delete StorageGRID Tenant Account
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), display_name=dict(required=False, type='str'), unique_name=dict(required=True, type='str'), management_policy=dict(required=False, type='dict', options=dict(manage_all_containers=dict(required=False, type='bool'), manage_endpoints=dict(required=False, type='bool'), manage_own_s3_credentials=dict(required=False, type='bool'), root_access=dict(required=False, type='bool'))), s3_policy=dict(required=False, type='json')))
        parameter_map = {'manage_all_containers': 'manageAllContainers', 'manage_endpoints': 'manageEndpoints', 'manage_own_s3_credentials': 'manageOwnS3Credentials', 'root_access': 'rootAccess'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        self.data['displayName'] = self.parameters.get('display_name')
        self.data['uniqueName'] = self.parameters['unique_name']
        self.data['policies'] = {}
        if self.parameters.get('management_policy'):
            self.data['policies'] = {'management': dict(((parameter_map[k], v) for k, v in self.parameters['management_policy'].items() if v))}
        if not self.data['policies'].get('management'):
            self.data['policies']['management'] = None
        if self.parameters.get('s3_policy'):
            try:
                self.data['policies']['s3'] = json.loads(self.parameters['s3_policy'])
            except ValueError:
                self.module.fail_json(msg='Failed to decode s3_policy. Invalid JSON.')
        self.re_local_group = re.compile('^group/')
        self.re_fed_group = re.compile('^federated-group/')
        if self.re_local_group.match(self.parameters['unique_name']) is None and self.re_fed_group.match(self.parameters['unique_name']) is None:
            self.module.fail_json(msg="unique_name must begin with 'group/' or 'federated-group/'")

    def get_org_group(self, unique_name):
        api = 'api/v3/org/groups/%s' % unique_name
        response, error = self.rest_api.get(api)
        if error:
            if response['code'] != 404:
                self.module.fail_json(msg=error)
        else:
            return response['data']
        return None

    def create_org_group(self):
        api = 'api/v3/org/groups'
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_org_group(self, group_id):
        api = 'api/v3/org/groups/' + group_id
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def update_org_group(self, group_id):
        api = 'api/v3/org/groups/' + group_id
        response, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        org_group = self.get_org_group(self.parameters['unique_name'])
        cd_action = self.na_helper.get_cd_action(org_group, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            if self.parameters.get('management_policy'):
                if org_group.get('policies') is None or org_group.get('policies', {}).get('management') != self.data['policies']['management']:
                    update = True
            if self.parameters.get('s3_policy'):
                if org_group.get('policies') is None or org_group.get('policies', {}).get('s3') != self.data['policies']['s3']:
                    update = True
            if update:
                self.na_helper.changed = True
        result_message = ''
        resp_data = org_group
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'delete':
                self.delete_org_group(org_group['id'])
                result_message = 'Org Group deleted'
            elif cd_action == 'create':
                resp_data = self.create_org_group()
                result_message = 'Org Group created'
            else:
                if self.re_fed_group.match(self.parameters['unique_name']):
                    self.data['displayName'] = org_group['displayName']
                resp_data = self.update_org_group(org_group['id'])
                result_message = 'Org Group updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)