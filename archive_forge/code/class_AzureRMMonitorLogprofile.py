from __future__ import absolute_import, division, print_function
class AzureRMMonitorLogprofile(AzureRMModuleBase):
    """Configuration class for an Azure RM Monitor log profile"""

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), location=dict(type='str'), locations=dict(type='list', elements='str'), categories=dict(type='list', elements='str'), retention_policy=dict(type='dict', options=retention_policy_spec), service_bus_rule_id=dict(type='str'), storage_account=dict(type='raw'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self._client = None
        self.name = None
        self.location = None
        self.locations = None
        self.categories = None
        self.retention_policy = False
        self.service_bus_rule_id = None
        self.storage_account = None
        self.tags = None
        self.results = dict(changed=False, id=None)
        self.state = None
        super(AzureRMMonitorLogprofile, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        old_response = None
        response = None
        to_be_updated = False
        if self.storage_account:
            if isinstance(self.storage_account, dict):
                self.storage_account = format_resource_id(val=self.storage_account['name'], subscription_id=self.storage_account.get('subscription') or self.subscription_id, namespace='Microsoft.Storage', types='storageAccounts', resource_group=self.storage_account.get('resource_group'))
            elif not is_valid_resource_id(self.storage_account):
                self.fail('storage_account either be a resource id or a dict containing resource_group and name')
        old_response = self.get_logprofile()
        if old_response:
            self.results['id'] = old_response['id']
        if self.state == 'present':
            if not old_response:
                self.log("Log profile instance doesn't exist")
                to_be_updated = True
                self.to_do = Actions.CreateOrUpdate
            else:
                self.log('Log profile instance already exists')
                update_tags, self.tags = self.update_tags(old_response.get('tags', None))
                if update_tags:
                    to_be_updated = True
                    self.to_do = Actions.CreateOrUpdate
                if self.check_update(old_response):
                    to_be_updated = True
                    self.to_do = Actions.CreateOrUpdate
        elif self.state == 'absent':
            if old_response:
                self.log('Delete log profile instance')
                self.results['id'] = old_response['id']
                to_be_updated = True
                self.to_do = Actions.Delete
            else:
                self.results['changed'] = False
                self.log('Log profile {0} not exists.'.format(self.name))
        if to_be_updated:
            self.log('Need to Create/Update log profile')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            if self.to_do == Actions.CreateOrUpdate:
                response = self.create_or_update_logprofile()
                self.results['id'] = response['id']
            if self.to_do == Actions.Delete:
                self.delete_logprofile()
                self.log('Log profile instance deleted')
        return self.results

    def check_update(self, existing):
        if self.locations is not None and existing['locations'] != self.locations:
            self.log('locations diff: origin {0} / update {1}'.format(existing['locations'], self.locations))
            return True
        if self.retention_policy is not None:
            if existing['retention_policy']['enabled'] != self.retention_policy['enabled']:
                self.log('retention_policy diff: origin {0} / update {1}'.format(str(existing['sku']['name']), str(self.retention_policy['enabled'])))
                return True
            if existing['retention_policy']['days'] != self.retention_policy['days']:
                self.log('retention_policy diff: origin {0} / update {1}'.format(existing['retention_policy']['days'], str(self.retention_policy['days'])))
                return True
        if self.storage_account is not None and existing['storage_account'] != self.storage_account:
            self.log('storage_account diff: origin {0} / update {1}'.format(existing['storage_account'], self.storage_account))
            return True
        if self.service_bus_rule_id is not None and existing['service_bus_rule_id'] != self.service_bus_rule_id:
            self.log('service_bus_rule_id diff: origin {0} / update {1}'.format(existing['service_bus_rule_id'], self.service_bus_rule_id))
            return True
        return False

    def create_or_update_logprofile(self):
        """
        Creates or Update log profile.

        :return: deserialized log profile state dictionary
        """
        self.log('Creating log profile instance {0}'.format(self.name))
        try:
            params = LogProfileResource(location=self.location, locations=self.locations, categories=self.categories, retention_policy=RetentionPolicy(days=self.retention_policy['days'], enabled=self.retention_policy['enabled']) if self.retention_policy else None, storage_account_id=self.storage_account if self.storage_account else None, service_bus_rule_id=self.service_bus_rule_id if self.service_bus_rule_id else None, tags=self.tags)
            response = self.monitor_log_profiles_client.log_profiles.create_or_update(log_profile_name=self.name, parameters=params)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except HttpResponseError as exc:
            self.log('Error attempting to create/update log profile.')
            self.fail('Error creating/updating log profile: {0}'.format(str(exc)))
        return logprofile_to_dict(response)

    def delete_logprofile(self):
        """
        Deletes specified log profile.

        :return: True
        """
        self.log('Deleting the log profile instance {0}'.format(self.name))
        try:
            response = self.monitor_log_profiles_client.log_profiles.delete(log_profile_name=self.name)
        except HttpResponseError as e:
            self.log('Error attempting to delete the log profile.')
            self.fail('Error deleting the log profile: {0}'.format(str(e)))
        return True

    def get_logprofile(self):
        """
        Gets the properties of the specified log profile.

        :return: log profile state dictionary
        """
        self.log('Checking if the log profile {0} is present'.format(self.name))
        response = None
        try:
            response = self.monitor_log_profiles_client.log_profiles.get(log_profile_name=self.name)
            self.log('Response : {0}'.format(response))
            self.log('log profile : {0} found'.format(response.name))
            return logprofile_to_dict(response)
        except HttpResponseError:
            self.log("Didn't find log profile {0}".format(self.name))
        return False