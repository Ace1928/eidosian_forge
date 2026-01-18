from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
class NetAppCloudmanagerInfo(object):
    """
    Contains methods to parse arguments,
    derive details of CloudmanagerInfo objects
    and send requests to CloudmanagerInfo via
    the restApi
    """

    def __init__(self):
        self.argument_spec = netapp_utils.cloudmanager_host_argument_spec()
        self.argument_spec.update(dict(gather_subsets=dict(type='list', elements='str', default='all'), client_id=dict(required=True, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_one_of=[['refresh_token', 'sa_client_id']], required_together=[['sa_client_id', 'sa_secret_key']], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = CloudManagerRestAPI(self.module)
        self.rest_api.url += self.rest_api.environment_data['CLOUD_MANAGER_HOST']
        self.rest_api.api_root_path = None
        self.methods = dict(working_environments_info=self.na_helper.get_working_environments_info, aggregates_info=self.get_aggregates_info, accounts_info=self.na_helper.get_accounts_info, account_info=self.na_helper.get_account_info, agents_info=self.na_helper.get_agents_info, active_agents_info=self.na_helper.get_active_agents_info)
        self.headers = {}
        if 'client_id' in self.parameters:
            self.headers['X-Agent-Id'] = self.rest_api.format_client_id(self.parameters['client_id'])

    def get_aggregates_info(self, rest_api, headers):
        """
        Get aggregates info: there are 4 types of working environments.
        Each of the aggregates will be categorized by working environment type and working environment id
        """
        aggregates = {}
        working_environments, error = self.na_helper.get_working_environments_info(rest_api, headers)
        if error is not None:
            self.module.fail_json(msg='Error: Failed to get working environments: %s' % str(error))
        for working_env_type in working_environments:
            we_aggregates = {}
            for we in working_environments[working_env_type]:
                provider = we['cloudProviderName']
                working_environment_id = we['publicId']
                self.na_helper.set_api_root_path(we, rest_api)
                if provider != 'Amazon':
                    api = '%s/aggregates/%s' % (rest_api.api_root_path, working_environment_id)
                else:
                    api = '%s/aggregates?workingEnvironmentId=%s' % (rest_api.api_root_path, working_environment_id)
                response, error, dummy = rest_api.get(api, None, header=headers)
                if error:
                    self.module.fail_json(msg='Error: Failed to get aggregate list: %s' % str(error))
                we_aggregates[working_environment_id] = response
            aggregates[working_env_type] = we_aggregates
        return aggregates

    def get_info(self, func, rest_api):
        """
        Main get info function
        """
        return self.methods[func](rest_api, self.headers)

    def apply(self):
        """
        Apply action to the Cloud Manager
        :return: None
        """
        info = {}
        if 'all' in self.parameters['gather_subsets']:
            self.parameters['gather_subsets'] = self.methods.keys()
        for func in self.parameters['gather_subsets']:
            if func in self.methods:
                info[func] = self.get_info(func, self.rest_api)
            else:
                msg = '%s is not a valid gather_subset. Only %s are allowed' % (func, self.methods.keys())
                self.module.fail_json(msg=msg)
        self.module.exit_json(changed=False, info=info)