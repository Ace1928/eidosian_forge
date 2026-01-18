from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMArtifactInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), artifact_source_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.artifact_source_name = None
        self.name = None
        super(AzureRMArtifactInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['artifacts'] = self.get()
        else:
            self.results['artifacts'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.artifacts.get(resource_group_name=self.resource_group, lab_name=self.lab_name, artifact_source_name=self.artifact_source_name, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Artifact.')
        if response:
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.artifacts.list(resource_group_name=self.resource_group, lab_name=self.lab_name, artifact_source_name=self.artifact_source_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Artifact.')
        if response is not None:
            for item in response:
                results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.parse_resource_to_dict(d.get('id')).get('resource_group'), 'lab_name': self.parse_resource_to_dict(d.get('id')).get('name'), 'artifact_source_name': self.parse_resource_to_dict(d.get('id')).get('child_name_1'), 'id': d.get('id'), 'description': d.get('description'), 'file_path': d.get('file_path'), 'name': d.get('name'), 'parameters': d.get('parameters'), 'publisher': d.get('publisher'), 'target_os_type': d.get('target_os_type'), 'title': d.get('title')}
        return d