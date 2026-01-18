from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDeploymentInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        super(AzureRMDeploymentInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_deployment_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_deployment_facts' module has been renamed to 'azure_rm_deployment_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name:
            self.results['deployments'] = self.get()
        else:
            self.results['deployments'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.rm_client.deployments.get(self.resource_group, deployment_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Deployment.')
        if response:
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.rm_client.deployments.list_by_resource_group(self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Deployment.')
        if response is not None:
            for item in response:
                results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        output_resources = {}
        for dependency in d.get('properties', {}).get('dependencies'):
            depends_on = []
            for depends_on_resource in dependency['depends_on']:
                depends_on.append(depends_on_resource['id'])
                if not output_resources.get(depends_on_resource['id']):
                    sub_resource = {'id': depends_on_resource['id'], 'name': depends_on_resource['resource_name'], 'type': depends_on_resource['resource_type'], 'depends_on': []}
                    output_resources[depends_on_resource['id']] = sub_resource
            resource = {'id': dependency['id'], 'name': dependency['resource_name'], 'type': dependency['resource_type'], 'depends_on': depends_on}
            output_resources[dependency['id']] = resource
        output_resources_list = []
        for r in output_resources:
            output_resources_list.append(output_resources[r])
        d = {'id': d.get('id'), 'resource_group': self.resource_group, 'name': d.get('name'), 'provisioning_state': d.get('properties', {}).get('provisioning_state'), 'parameters': d.get('properties', {}).get('parameters'), 'outputs': d.get('properties', {}).get('outputs'), 'output_resources': output_resources_list, 'template_link': d.get('properties', {}).get('template_link', {}).get('uri'), 'correlation_id': d.get('properties', {}).get('correlation_id')}
        return d