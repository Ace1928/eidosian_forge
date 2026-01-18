from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAppServicePlans(AzureRMModuleBase):
    """Configuration class for an Azure RM App Service Plan resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), sku=dict(type='str'), is_linux=dict(type='bool', default=False), number_of_workers=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.sku = None
        self.is_linux = None
        self.number_of_workers = 1
        self.tags = None
        self.results = dict(changed=False, ansible_facts=dict(azure_appserviceplan=None))
        self.state = None
        super(AzureRMAppServicePlans, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if kwargs[key]:
                setattr(self, key, kwargs[key])
        old_response = None
        response = None
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        old_response = self.get_plan()
        if not old_response:
            self.log("App Service plan doesn't exist")
            if self.state == 'present':
                to_be_updated = True
                if not self.sku:
                    self.fail('Please specify sku in plan when creation')
        else:
            self.log('App Service Plan already exists')
            if self.state == 'present':
                self.log('Result: {0}'.format(old_response))
                update_tags, newtags = self.update_tags(old_response.get('tags', dict()))
                if update_tags:
                    to_be_updated = True
                    self.tags = newtags
                if self.sku and _normalize_sku(self.sku) != _normalize_sku(old_response['sku']['size']):
                    to_be_updated = True
                if self.number_of_workers and int(self.number_of_workers) != old_response['sku']['capacity']:
                    to_be_updated = True
                if self.is_linux and self.is_linux != old_response['reserved']:
                    self.fail('Operation not allowed: cannot update reserved of app service plan.')
        if old_response:
            self.results['id'] = old_response['id']
        if to_be_updated:
            self.log('Need to Create/Update app service plan')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_or_update_plan()
            self.results['id'] = response['id']
        if self.state == 'absent' and old_response:
            self.log('Delete app service plan')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_plan()
            self.log('App service plan instance deleted')
        return self.results

    def get_plan(self):
        """
        Gets app service plan
        :return: deserialized app service plan dictionary
        """
        self.log('Get App Service Plan {0}'.format(self.name))
        try:
            response = self.web_client.app_service_plans.get(resource_group_name=self.resource_group, name=self.name)
            if response:
                self.log('Response : {0}'.format(response))
                self.log('App Service Plan : {0} found'.format(response.name))
                return appserviceplan_to_dict(response)
        except ResourceNotFoundError:
            self.log("Didn't find app service plan {0} in resource group {1}".format(self.name, self.resource_group))
        return False

    def create_or_update_plan(self):
        """
        Creates app service plan
        :return: deserialized app service plan dictionary
        """
        self.log('Create App Service Plan {0}'.format(self.name))
        try:
            sku = _normalize_sku(self.sku)
            sku_def = SkuDescription(tier=get_sku_name(sku), name=sku, capacity=self.number_of_workers)
            plan_def = AppServicePlan(location=self.location, app_service_plan_name=self.name, sku=sku_def, reserved=self.is_linux, tags=self.tags if self.tags else None)
            response = self.web_client.app_service_plans.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, app_service_plan=plan_def)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            self.log('Response : {0}'.format(response))
            return appserviceplan_to_dict(response)
        except Exception as ex:
            self.fail('Failed to create app service plan {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def delete_plan(self):
        """
        Deletes specified App service plan in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the App service plan {0}'.format(self.name))
        try:
            self.web_client.app_service_plans.delete(resource_group_name=self.resource_group, name=self.name)
        except ResourceNotFoundError as e:
            self.log('Error attempting to delete App service plan.')
            self.fail('Error deleting the App service plan : {0}'.format(str(e)))
        return True