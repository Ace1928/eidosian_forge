from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMContainerRegistryTagInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(registry=dict(type='str', required=True), repository_name=dict(type='str'), name=dict(type='str'))
        self.results = dict(changed=False)
        self.registry = None
        self.repository_name = None
        self.name = None
        self._client = None
        super(AzureRMContainerRegistryTagInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self._client = self.get_client()
        if self.repository_name and self.name:
            self.results['repositories'] = [self.get_tag(self.repository_name, self.name)]
        elif self.repository_name:
            tags = self.list_by_repository(self.repository_name, self.name)
            self.results['repositories'] = [] if not tags else [tags]
        else:
            self.results['repositories'] = self.list_all_repositories(self.name)
        return self.results

    def get_client(self):
        registry_endpoint = self.registry if self.registry.endswith('.azurecr.io') else self.registry + '.azurecr.io'
        return ContainerRegistryClient(endpoint=registry_endpoint, credential=self.azure_auth.azure_credential_track2, audience='https://management.azure.com')

    def get_tag(self, repository_name, tag_name):
        response = None
        try:
            response = self._client.get_tag_properties(repository=repository_name, tag=tag_name)
            self.log(f'Response : {response}')
        except Exception as e:
            self.log(f'Could not get ACR tag for {repository_name}:{tag_name} - {str(e)}')
        tags = []
        if response is not None:
            tags.append(format_tag(response))
        return {'name': repository_name, 'tags': tags}

    def list_by_repository(self, repository_name, tag_name):
        try:
            response = self._client.list_tag_properties(repository=repository_name)
            self.log(f'Response : {response}')
            tags = []
            for tag in response:
                if not tag_name or tag.name == tag_name:
                    tags.append(format_tag(tag))
            return {'name': repository_name, 'tags': tags}
        except ResourceNotFoundError as e:
            self.log(f'Could not get ACR tags for {repository_name} - {str(e)}')
        return None

    def list_all_repositories(self, tag_name):
        response = None
        try:
            response = self._client.list_repository_names()
            self.log(f'Response : {response}')
        except Exception as e:
            self.fail(f'Could not get ACR repositories - {str(e)}')
        if response is not None:
            results = []
            for repo_name in response:
                tags = self.list_by_repository(repo_name, tag_name)
                if tags:
                    results.append(tags)
            return results
        return None