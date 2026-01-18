from __future__ import absolute_import, division, print_function
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMApiManagementServiceInfo(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'))
        self.resource_group = None
        self.name = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200]
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2020-06-01-preview'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        self.mgmt_client = None
        super(AzureRMApiManagementServiceInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.resource_group is not None and self.name is not None:
            self.results['api_management_service'] = self.get()
        elif self.resource_group is not None:
            self.results['api_management_service'] = self.listbyresourcegroup()
        else:
            self.results['api_management_service'] = self.list()
        return self.results

    def get(self):
        response = None
        results = {}
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.ApiManagement' + '/service' + '/{{ service_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ service_name }}', self.name)
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            results = json.loads(response.body())
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return self.format_item(results)

    def listbyresourcegroup(self):
        response = None
        results = {}
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.ApiManagement' + '/service'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            results = json.loads(response.body())
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return [self.format_item(x) for x in results['value']] if results['value'] else []

    def list(self):
        response = None
        results = {}
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/providers' + '/Microsoft.ApiManagement' + '/service'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            results = json.loads(response.body())
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return [self.format_item(x) for x in results['value']] if results['value'] else []

    def format_item(self, item):
        if item:
            d = {'id': item['id'], 'name': item['name'], 'type': item['type'], 'sku': item['sku']['name'], 'identity': item['identity'], 'zones': item['zones'], 'location': item['location'], 'etag': item['etag'], 'properties': item['properties']}
        else:
            return dict()
        return d