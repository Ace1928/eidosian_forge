from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMDtlCustomImage(AzureRMModuleBase):
    """Configuration class for an Azure RM Custom Image resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), name=dict(type='str', required=True), source_vm=dict(type='str'), windows_os_state=dict(type='str', choices=['non_sysprepped', 'sysprep_requested', 'sysprep_applied']), linux_os_state=dict(type='str', choices=['non_deprovisioned', 'deprovision_requested', 'deprovision_applied']), description=dict(type='str'), author=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.lab_name = None
        self.name = None
        self.custom_image = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        required_if = [('state', 'present', ['source_vm'])]
        super(AzureRMDtlCustomImage, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.custom_image[key] = kwargs[key]
        if self.state == 'present':
            windows_os_state = self.custom_image.pop('windows_os_state', False)
            linux_os_state = self.custom_image.pop('linux_os_state', False)
            source_vm_name = self.custom_image.pop('source_vm')
            temp = '/subscriptions/{0}/resourcegroups/{1}/providers/microsoft.devtestlab/labs/{2}/virtualmachines/{3}'
            self.custom_image['vm'] = {}
            self.custom_image['vm']['source_vm_id'] = temp.format(self.subscription_id, self.resource_group, self.lab_name, source_vm_name)
            if windows_os_state:
                self.custom_image['vm']['windows_os_info'] = {'windows_os_state': _snake_to_camel(windows_os_state, True)}
            elif linux_os_state:
                self.custom_image['vm']['linux_os_info'] = {'linux_os_state': _snake_to_camel(linux_os_state, True)}
            else:
                self.fail("Either 'linux_os_state' or 'linux_os_state' must be specified")
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        old_response = self.get_customimage()
        if not old_response:
            self.log("Custom Image instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Custom Image instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                if not default_compare(self.custom_image, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Custom Image instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_customimage()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Custom Image instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_customimage()
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        else:
            self.log('Custom Image instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None)})
        return self.results

    def create_update_customimage(self):
        """
        Creates or updates Custom Image with the specified configuration.

        :return: deserialized Custom Image instance state dictionary
        """
        self.log('Creating / Updating the Custom Image instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.custom_images.begin_create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, custom_image=self.custom_image)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Custom Image instance.')
            self.fail('Error creating the Custom Image instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_customimage(self):
        """
        Deletes specified Custom Image instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Custom Image instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.custom_images.begin_delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Custom Image instance.')
            self.fail('Error deleting the Custom Image instance: {0}'.format(str(e)))
        return True

    def get_customimage(self):
        """
        Gets the properties of the specified Custom Image.

        :return: deserialized Custom Image instance state dictionary
        """
        self.log('Checking if the Custom Image instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.custom_images.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Custom Image instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Custom Image instance.')
        if found is True:
            return response.as_dict()
        return False