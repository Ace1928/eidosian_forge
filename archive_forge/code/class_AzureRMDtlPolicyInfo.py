from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDtlPolicyInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), policy_set_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.policy_set_name = None
        self.name = None
        self.tags = None
        super(AzureRMDtlPolicyInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlabpolicy_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlabpolicy_facts' module has been renamed to 'azure_rm_devtestlabpolicy_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['policies'] = self.get()
        else:
            self.results['policies'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.policies.get(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Policy.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.policies.list(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Policy.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'policy_set_name': self.policy_set_name, 'name': d.get('name'), 'id': d.get('id'), 'tags': d.get('tags'), 'status': d.get('status'), 'threshold': d.get('threshold'), 'fact_name': d.get('fact_name'), 'evaluator_type': d.get('evaluator_type')}
        return d