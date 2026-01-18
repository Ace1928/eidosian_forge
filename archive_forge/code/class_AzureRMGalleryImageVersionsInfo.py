from __future__ import absolute_import, division, print_function
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMGalleryImageVersionsInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), gallery_name=dict(type='str', required=True), gallery_image_name=dict(type='str', required=True), name=dict(type='str'))
        self.resource_group = None
        self.gallery_name = None
        self.gallery_image_name = None
        self.name = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200]
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2019-03-01'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        self.mgmt_client = None
        super(AzureRMGalleryImageVersionsInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.resource_group is not None and self.gallery_name is not None and (self.gallery_image_name is not None) and (self.name is not None):
            self.results['versions'] = self.get()
        elif self.resource_group is not None and self.gallery_name is not None and (self.gallery_image_name is not None):
            self.results['versions'] = self.listbygalleryimage()
        return self.results

    def get(self):
        response = None
        results = {}
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/galleries' + '/{{ gallery_name }}' + '/images' + '/{{ image_name }}' + '/versions' + '/{{ version_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ gallery_name }}', self.gallery_name)
        self.url = self.url.replace('{{ image_name }}', self.gallery_image_name)
        self.url = self.url.replace('{{ version_name }}', self.name)
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            results = json.loads(response.body())
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return self.format_item(results)

    def listbygalleryimage(self):
        response = None
        results = dict(response=[])
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/galleries' + '/{{ gallery_name }}' + '/images' + '/{{ image_name }}' + '/versions'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ gallery_name }}', self.gallery_name)
        self.url = self.url.replace('{{ image_name }}', self.gallery_image_name)
        try:
            skiptoken = None
            while True:
                if skiptoken:
                    self.query_parameters['skiptoken'] = skiptoken
                response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, [200, 404], 0, 0)
                try:
                    response = json.loads(response.body())
                    if isinstance(response, dict):
                        if response.get('value'):
                            results['response'] = results['response'] + response['value']
                            skiptoken = response.get('nextLink')
                        else:
                            results['response'] = results['response'] + [response]
                except Exception as e:
                    self.fail('Failed to parse response: ' + str(e))
                if not skiptoken:
                    break
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return [self.format_item(x) for x in results['response']] if results['response'] else []

    def format_item(self, item):
        d = {'id': item['id'], 'name': item['name'], 'location': item['location'], 'tags': item.get('tags'), 'publishing_profile': item['properties']['publishingProfile'], 'provisioning_state': item['properties']['provisioningState']}
        return d