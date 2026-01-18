from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDtlEnvironment(AzureRMModuleBase):
    """Configuration class for an Azure RM Environment resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), user_name=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), deployment_template=dict(type='raw'), deployment_parameters=dict(type='list', elements='dict', options=dict(name=dict(type='str'), value=dict(type='str'))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.lab_name = None
        self.user_name = None
        self.name = None
        self.dtl_environment = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMDtlEnvironment, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.dtl_environment[key] = kwargs[key]
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        deployment_template = self.dtl_environment.pop('deployment_template', None)
        if deployment_template:
            if isinstance(deployment_template, dict):
                if all((key in deployment_template for key in ('artifact_source_name', 'name'))):
                    tmp = '/subscriptions/{0}/resourcegroups/{1}/providers/microsoft.devtestlab/labs/{2}/artifactSources/{3}/armTemplates/{4}'
                    deployment_template = tmp.format(self.subscription_id, self.resource_group, self.lab_name, deployment_template['artifact_source_name'], deployment_template['name'])
            if not isinstance(deployment_template, str):
                self.fail('parameter error: expecting deployment_template to contain [artifact_source, name]')
            self.dtl_environment['deployment_properties'] = {}
            self.dtl_environment['deployment_properties']['arm_template_id'] = deployment_template
            self.dtl_environment['deployment_properties']['parameters'] = self.dtl_environment.pop('deployment_parameters', None)
        old_response = self.get_environment()
        if not old_response:
            self.log("Environment instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Environment instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                if not default_compare(self.dtl_environment, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Environment instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_environment()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Environment instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_environment()
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        else:
            self.log('Environment instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None)})
        return self.results

    def create_update_environment(self):
        """
        Creates or updates Environment with the specified configuration.

        :return: deserialized Environment instance state dictionary
        """
        self.log('Creating / Updating the Environment instance {0}'.format(self.name))
        try:
            if self.to_do == Actions.Create:
                response = self.mgmt_client.environments.begin_create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name, dtl_environment=self.dtl_environment)
            else:
                response = self.mgmt_client.environments.update(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name, dtl_environment=self.dtl_environment)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Environment instance.')
            self.fail('Error creating the Environment instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_environment(self):
        """
        Deletes specified Environment instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Environment instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.environments.begin_delete(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Environment instance.')
            self.fail('Error deleting the Environment instance: {0}'.format(str(e)))
        return True

    def get_environment(self):
        """
        Gets the properties of the specified Environment.

        :return: deserialized Environment instance state dictionary
        """
        self.log('Checking if the Environment instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.environments.get(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Environment instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Environment instance.')
        if found is True:
            return response.as_dict()
        return False