from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDevTestLabVirtualNetworkInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.name = None
        super(AzureRMDevTestLabVirtualNetworkInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlabvirtualnetwork_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlabvirtualnetwork_facts' module has been renamed to 'azure_rm_devtestlabvirtualnetwork_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['virtualnetworks'] = self.get()
        else:
            self.results['virtualnetworks'] = self.list()
        return self.results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.virtual_networks.list(resource_group_name=self.resource_group, lab_name=self.lab_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not list Virtual Networks for DevTest Lab.')
        if response is not None:
            for item in response:
                results.append(self.format_response(item))
        return results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.virtual_networks.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get facts for Virtual Network.')
        if response:
            results.append(self.format_response(response))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'lab_name': self.lab_name, 'name': d.get('name', None), 'id': d.get('id', None), 'external_provider_resource_id': d.get('external_provider_resource_id', None), 'provisioning_state': d.get('provisioning_state', None), 'description': d.get('description', None)}
        return d