from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSchedule(AzureRMModuleBase):
    """Configuration class for an Azure RM Schedule resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), name=dict(type='str', required=True, choices=['lab_vms_startup', 'lab_vms_shutdown']), time=dict(type='str'), time_zone_id=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.lab_name = None
        self.name = None
        self.schedule = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        required_if = [('state', 'present', ['time', 'time_zone_id'])]
        super(AzureRMSchedule, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.schedule[key] = kwargs[key]
        self.schedule['status'] = 'Enabled'
        if self.name == 'lab_vms_startup':
            self.name = 'LabVmsStartup'
            self.schedule['task_type'] = 'LabVmsStartupTask'
        elif self.name == 'lab_vms_shutdown':
            self.name = 'LabVmsShutdown'
            self.schedule['task_type'] = 'LabVmsShutdownTask'
        if self.state == 'present':
            self.schedule['daily_recurrence'] = {'time': self.schedule.pop('time')}
            self.schedule['time_zone_id'] = self.schedule['time_zone_id'].upper()
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        old_response = self.get_schedule()
        if not old_response:
            self.log("Schedule instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Schedule instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                if not default_compare(self.schedule, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Schedule instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_schedule()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Schedule instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_schedule()
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        else:
            self.log('Schedule instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None)})
        return self.results

    def create_update_schedule(self):
        """
        Creates or updates Schedule with the specified configuration.

        :return: deserialized Schedule instance state dictionary
        """
        self.log('Creating / Updating the Schedule instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.schedules.create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, schedule=self.schedule)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Schedule instance.')
            self.fail('Error creating the Schedule instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_schedule(self):
        """
        Deletes specified Schedule instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Schedule instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.schedules.delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Schedule instance.')
            self.fail('Error deleting the Schedule instance: {0}'.format(str(e)))
        return True

    def get_schedule(self):
        """
        Gets the properties of the specified Schedule.

        :return: deserialized Schedule instance state dictionary
        """
        self.log('Checking if the Schedule instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.schedules.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Schedule instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Schedule instance.')
        if found is True:
            return response.as_dict()
        return False