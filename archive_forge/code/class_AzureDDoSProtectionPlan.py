from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureDDoSProtectionPlan(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.resource_group = None
        self.name = None
        self.location = None
        self.state = None
        self.tags = None
        self.log_path = None
        self.results = dict(changed=False, state=dict())
        super(AzureDDoSProtectionPlan, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results['check_mode'] = self.check_mode
        self.get_resource_group(self.resource_group)
        results = dict()
        changed = False
        try:
            self.log('Fetching DDoS protection plan {0}'.format(self.name))
            ddos_protection_plan = self.network_client.ddos_protection_plans.get(self.resource_group, self.name)
            results = ddos_protection_plan_to_dict(ddos_protection_plan)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                self.results['state'] = self.create_or_update_ddos_protection_plan(self.module.params)
            elif self.state == 'absent':
                self.delete_ddos_protection_plan()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_ddos_protection_plan(self, params):
        """
        Create or update DDoS protection plan.
        :return: create or update DDoS protection plan instance state dictionary
        """
        self.log('create or update DDoS protection plan {0}'.format(self.name))
        try:
            poller = self.network_client.ddos_protection_plans.begin_create_or_update(resource_group_name=params.get('resource_group'), ddos_protection_plan_name=params.get('name'), parameters=params)
            result = self.get_poller_result(poller)
            self.log('Response : {0}'.format(result))
        except Exception as ex:
            self.fail('Failed to create DDoS protection plan {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return ddos_protection_plan_to_dict(result)

    def delete_ddos_protection_plan(self):
        """
        Deletes specified DDoS protection plan
        :return True
        """
        self.log('Deleting the DDoS protection plan {0}'.format(self.name))
        try:
            poller = self.network_client.ddos_protection_plans.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except ResourceNotFoundError as e:
            self.log('Error attempting to delete DDoS protection plan.')
            self.fail('Error deleting the DDoS protection plan : {0}'.format(str(e)))
        return result