from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMRegistrationAssignmentInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(scope=dict(type='str', required=True), registration_assignment_id=dict(type='str'))
        self.scope = None
        self.registration_assignment_id = None
        self.expand_registration_definition = False
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200]
        self.mgmt_client = None
        super(AzureRMRegistrationAssignmentInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(ManagedServicesClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2020-09-01', is_track2=True, suppress_subscription_id=True)
        if self.scope is not None and self.registration_assignment_id is not None:
            self.results['registration_assignments'] = self.format_item(self.get())
        elif self.scope is not None:
            self.results['registration_assignments'] = self.format_item(self.list())
        if len(self.results['registration_assignments']) > 0:
            for item in self.results['registration_assignments']:
                if item.get('properties', None) is not None:
                    registration_definition_id = item['properties']['registration_definition_id']
                    item['properties'].clear()
                    item['properties']['registration_definition_id'] = registration_definition_id
        return self.results

    def get(self):
        response = None
        try:
            response = self.mgmt_client.registration_assignments.get(scope=self.scope, registration_assignment_id=self.registration_assignment_id, expand_registration_definition=self.expand_registration_definition)
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list(self):
        response = None
        try:
            response = self.mgmt_client.registration_assignments.list(scope=self.scope, expand_registration_definition=self.expand_registration_definition)
        except Exception as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if hasattr(item, 'as_dict'):
            return [item.as_dict()]
        else:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result