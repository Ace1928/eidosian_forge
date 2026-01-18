from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAppServicePlanInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.tags = None
        self.info_level = None
        super(AzureRMAppServicePlanInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_appserviceplan_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_appserviceplan_facts' module has been renamed to 'azure_rm_appserviceplan_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name:
            self.results['appserviceplans'] = self.list_by_name()
        elif self.resource_group:
            self.results['appserviceplans'] = self.list_by_resource_group()
        else:
            self.results['appserviceplans'] = self.list_all()
        return self.results

    def list_by_name(self):
        self.log('Get app service plan {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.web_client.app_service_plans.get(resource_group_name=self.resource_group, name=self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            curated_result = self.construct_curated_plan(item)
            result = [curated_result]
        return result

    def list_by_resource_group(self):
        self.log('List app service plans in resource groups {0}'.format(self.resource_group))
        try:
            response = list(self.web_client.app_service_plans.list_by_resource_group(resource_group_name=self.resource_group))
        except Exception as exc:
            self.fail('Error listing app service plan in resource groups {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                curated_output = self.construct_curated_plan(item)
                results.append(curated_output)
        return results

    def list_all(self):
        self.log('List app service plans in current subscription')
        try:
            response = list(self.web_client.app_service_plans.list())
        except Exception as exc:
            self.fail('Error listing app service plans: {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                curated_output = self.construct_curated_plan(item)
                results.append(curated_output)
        return results

    def construct_curated_plan(self, plan):
        plan_facts = self.serialize_obj(plan, AZURE_OBJECT_CLASS)
        curated_output = dict()
        curated_output['id'] = plan_facts['id']
        curated_output['name'] = plan_facts['name']
        curated_output['resource_group'] = plan_facts['resource_group']
        curated_output['location'] = plan_facts['location']
        curated_output['tags'] = plan_facts.get('tags', None)
        curated_output['is_linux'] = False
        curated_output['kind'] = plan_facts['kind']
        curated_output['sku'] = plan_facts['sku']
        if plan_facts.get('reserved', None):
            curated_output['is_linux'] = True
        return curated_output