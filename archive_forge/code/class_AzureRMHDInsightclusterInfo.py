from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMHDInsightclusterInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMHDInsightclusterInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_hdinsightcluster_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_hdinsightcluster_facts' module has been renamed to 'azure_rm_hdinsightcluster_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(HDInsightManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name is not None:
            self.results['clusters'] = self.get()
        elif self.resource_group is not None:
            self.results['clusters'] = self.list_by_resource_group()
        else:
            self.results['clusters'] = self.list_all()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.clusters.get(resource_group_name=self.resource_group, cluster_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for HDInsight Cluster.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.clusters.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for HDInsight Cluster.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def list_all(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.clusters.list()
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for HDInsight Cluster.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'id': d.get('id'), 'resource_group': self.parse_resource_to_dict(d.get('id')).get('resource_group'), 'name': d.get('name', None), 'location': d.get('location', '').replace(' ', '').lower(), 'cluster_version': d.get('properties', {}).get('cluster_version'), 'os_type': d.get('properties', {}).get('os_type'), 'tier': d.get('properties', {}).get('tier'), 'cluster_definition': {'kind': d.get('properties', {}).get('cluster_definition', {}).get('kind')}, 'compute_profile_roles': [{'name': item.get('name'), 'target_instance_count': item.get('target_instance_count'), 'vm_size': item.get('hardware_profile', {}).get('vm_size'), 'linux_profile': {'username': item.get('os_profile', {}).get('linux_operating_system_profile', {}).get('username')}} for item in d.get('properties', []).get('compute_profile', {}).get('roles', [])], 'connectivity_endpoints': d.get('properties', {}).get('connectivity_endpoints'), 'tags': d.get('tags', None)}
        return d