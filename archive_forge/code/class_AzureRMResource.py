from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible.module_utils.common.dict_transformations import dict_merge
class AzureRMResource(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(url=dict(type='str'), provider=dict(type='str'), resource_group=dict(type='str'), resource_type=dict(type='str'), resource_name=dict(type='str'), subresource=dict(type='list', elements='dict', default=[]), api_version=dict(type='str'), method=dict(type='str', default='PUT', choices=['GET', 'PUT', 'POST', 'HEAD', 'PATCH', 'DELETE', 'MERGE']), body=dict(type='raw'), status_code=dict(type='list', elements='int', default=[200, 201, 202]), idempotency=dict(type='bool', default=False), polling_timeout=dict(type='int', default=0), polling_interval=dict(type='int', default=60), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.results = dict(changed=False, response=None)
        self.mgmt_client = None
        self.url = None
        self.api_version = None
        self.provider = None
        self.resource_group = None
        self.resource_type = None
        self.resource_name = None
        self.subresource_type = None
        self.subresource_name = None
        self.subresource = []
        self.method = None
        self.status_code = []
        self.idempotency = False
        self.polling_timeout = None
        self.polling_interval = None
        self.state = None
        self.body = None
        super(AzureRMResource, self).__init__(self.module_arg_spec, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.state == 'absent':
            self.method = 'DELETE'
            self.status_code.append(204)
        if self.url is None:
            orphan = None
            rargs = dict()
            rargs['subscription'] = self.subscription_id
            rargs['resource_group'] = self.resource_group
            if not (self.provider is None or self.provider.lower().startswith('.microsoft')):
                rargs['namespace'] = 'Microsoft.' + self.provider
            else:
                rargs['namespace'] = self.provider
            if self.resource_type is not None and self.resource_name is not None:
                rargs['type'] = self.resource_type
                rargs['name'] = self.resource_name
                for i in range(len(self.subresource)):
                    resource_ns = self.subresource[i].get('namespace', None)
                    resource_type = self.subresource[i].get('type', None)
                    resource_name = self.subresource[i].get('name', None)
                    if resource_type is not None and resource_name is not None:
                        rargs['child_namespace_' + str(i + 1)] = resource_ns
                        rargs['child_type_' + str(i + 1)] = resource_type
                        rargs['child_name_' + str(i + 1)] = resource_name
                    else:
                        orphan = resource_type
            else:
                orphan = self.resource_type
            self.url = resource_id(**rargs)
            if orphan is not None:
                self.url += '/' + orphan
        if not self.api_version:
            try:
                if '/providers/' in self.url:
                    provider = self.url.split('/providers/')[1].split('/')[0]
                    resourceType = self.url.split(provider + '/')[1].split('/')[0]
                    url = '/subscriptions/' + self.subscription_id + '/providers/' + provider
                    api_versions = json.loads(self.mgmt_client.query(url, 'GET', {'api-version': '2015-01-01'}, None, None, [200], 0, 0).body())
                    for rt in api_versions['resourceTypes']:
                        if rt['resourceType'].lower() == resourceType.lower():
                            self.api_version = rt['apiVersions'][0]
                            break
                else:
                    self.api_version = '2018-05-01'
                if not self.api_version:
                    self.fail("Couldn't find api version for {0}/{1}".format(provider, resourceType))
            except Exception as exc:
                self.fail('Failed to obtain API version: {0}'.format(str(exc)))
        query_parameters = {}
        query_parameters['api-version'] = self.api_version
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        needs_update = True
        response = None
        if self.idempotency:
            original = self.mgmt_client.query(self.url, 'GET', query_parameters, None, None, [200, 404], 0, 0)
            if original.status_code == 404:
                if self.state == 'absent':
                    needs_update = False
            else:
                try:
                    response = json.loads(original.body())
                    needs_update = dict_merge(response, self.body) != response
                except Exception:
                    pass
        if needs_update:
            response = self.mgmt_client.query(self.url, self.method, query_parameters, header_parameters, self.body, self.status_code, self.polling_timeout, self.polling_interval)
            if self.state == 'present' and self.method != 'DELETE':
                try:
                    response = json.loads(response.body())
                except Exception:
                    response = response.context['deserialized_data']
            else:
                response = None
        self.results['response'] = response
        self.results['changed'] = needs_update
        return self.results