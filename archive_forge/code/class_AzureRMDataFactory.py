from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDataFactory(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), resource_group=dict(type='str', required=True), if_match=dict(type='str'), location=dict(type='str'), public_network_access=dict(type='str', choices=['Enabled', 'Disabled']), state=dict(type='str', default='present', choices=['absent', 'present']), repo_configuration=dict(type='dict', options=repo_configuration_spec))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.if_match = None
        self.location = None
        self.tags = None
        self.public_network_access = None
        self.repo_configuration = None
        super(AzureRMDataFactory, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True, facts_module=False)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        response = self.get_item()
        changed = False
        if self.state == 'present':
            if response:
                if self.tags is not None:
                    update_tags, tags = self.update_tags(response['tags'])
                    if update_tags:
                        changed = True
                        self.tags = tags
                if self.public_network_access is not None and self.public_network_access != response['public_network_access']:
                    changed = True
                else:
                    self.public_network_access = response['public_network_access']
                if self.repo_configuration is not None and self.repo_configuration != response['repo_configuration']:
                    changed = True
                else:
                    self.repo_configuration = response['repo_configuration']
            else:
                changed = True
            if self.check_mode:
                changed = True
                self.log('Check mode test, Data factory will be create or update')
            elif changed:
                if self.repo_configuration:
                    if self.repo_configuration['type'] == 'FactoryGitHubConfiguration':
                        repo_parameters = self.datafactory_model.FactoryGitHubConfiguration(account_name=self.repo_configuration.get('account_name'), repository_name=self.repo_configuration.get('repository_name'), collaboration_branch=self.repo_configuration.get('collaboration_branch'), root_folder=self.repo_configuration.get('root_folder'))
                    else:
                        repo_parameters = self.datafactory_model.FactoryVSTSConfiguration(account_name=self.repo_configuration.get('account_name'), repository_name=self.repo_configuration.get('repository_name'), collaboration_branch=self.repo_configuration.get('collaboration_branch'), root_folder=self.repo_configuration.get('root_folder'), project_name=self.repo_configuration.get('project_name'))
                else:
                    repo_parameters = None
                update_parameters = self.datafactory_model.Factory(location=self.location, tags=self.tags, public_network_access=self.public_network_access, repo_configuration=repo_parameters)
                response = self.create_or_update(update_parameters)
        else:
            if self.check_mode:
                changed = True
                self.log('Check mode test')
            if response:
                self.log('The Data factory {0} exist, will be deleted'.format(self.name))
                changed = True
                response = self.delete()
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = response
        return self.results

    def get_item(self):
        response = None
        self.log('Get properties for {0}'.format(self.name))
        try:
            response = self.datafactory_client.factories.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        return self.pip_to_dict(response) if response else None

    def delete(self):
        response = None
        self.log('Delete data factory for {0}'.format(self.name))
        try:
            response = self.datafactory_client.factories.delete(self.resource_group, self.name)
        except Exception as ec:
            self.fail('Delete fail {0}, error message {1}'.format(self.name, ec))
        return self.pip_to_dict(response) if response else None

    def create_or_update(self, parameters):
        response = None
        self.log('Create data factory for {0}'.format(self.name))
        try:
            response = self.datafactory_client.factories.create_or_update(self.resource_group, self.name, parameters, self.if_match)
        except Exception as ec:
            self.fail('Create fail {0}, error message {1}'.format(self.name, ec))
        return self.pip_to_dict(response) if response else None

    def pip_to_dict(self, pip):
        result = dict(id=pip.id, name=pip.name, type=pip.type, location=pip.location, tags=pip.tags, e_tag=pip.e_tag, provisioning_state=pip.provisioning_state, create_time=pip.create_time, public_network_access=pip.public_network_access, repo_configuration=dict(), identity=dict())
        if pip.identity:
            result['identity']['principal_id'] = pip.identity.principal_id
            result['identity']['tenant_id'] = pip.identity.tenant_id
        if pip.repo_configuration:
            result['repo_configuration']['account_name'] = pip.repo_configuration.account_name
            result['repo_configuration']['repository_name'] = pip.repo_configuration.repository_name
            result['repo_configuration']['collaboration_branch'] = pip.repo_configuration.collaboration_branch
            result['repo_configuration']['root_folder'] = pip.repo_configuration.root_folder
            result['repo_configuration']['type'] = pip.repo_configuration.type
            if pip.repo_configuration.type == 'FactoryVSTSConfiguration':
                result['repo_configuration']['project_name'] = pip.repo_configuration.project_name
        return result