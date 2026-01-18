from __future__ import absolute_import, division, print_function
import uuid
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMRegistrationDefinition(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(scope=dict(type='str'), registration_definition_id=dict(type='str'), properties=dict(type='dict', disposition='/properties', options=dict(description=dict(type='str', disposition='description'), authorizations=dict(type='list', disposition='authorizations', required=True, elements='dict', options=dict(principal_id=dict(type='str', disposition='principal_id', required=True), role_definition_id=dict(type='str', disposition='role_definition_id', required=True))), registration_definition_name=dict(type='str', disposition='registration_definition_name'), managed_by_tenant_id=dict(type='str', disposition='managed_by_tenant_id', required=True))), plan=dict(type='dict', disposition='/plan', options=dict(name=dict(type='str', disposition='name', required=True), publisher=dict(type='str', disposition='publisher', required=True), product=dict(type='str', disposition='product', required=True), version=dict(type='str', disposition='version', required=True))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.scope = None
        self.registration_definition_id = None
        self.body = {}
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMRegistrationDefinition, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        if self.registration_definition_id is None:
            self.registration_definition_id = str(uuid.uuid4())
        if not self.scope:
            self.scope = '/subscriptions/' + self.subscription_id
        else:
            self.scope = '/subscriptions/' + self.scope
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(ManagedServicesClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2019-09-01', is_track2=True, suppress_subscription_id=True)
        old_response = self.get_resource()
        if not old_response:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            modifiers = {}
            self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
            self.results['modifiers'] = modifiers
            self.results['compare'] = []
            if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_resource()
            self.results['state'] = response
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
        else:
            self.results['changed'] = False
            response = old_response
        return self.results

    def create_update_resource(self):
        try:
            response = self.mgmt_client.registration_definitions.begin_create_or_update(registration_definition_id=self.registration_definition_id, scope=self.scope, request_body=self.body)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the RegistrationDefinition instance.')
            self.fail('Error creating the RegistrationDefinition instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_resource(self):
        try:
            response = self.mgmt_client.registration_definitions.delete(registration_definition_id=self.registration_definition_id, scope=self.scope)
        except Exception as e:
            self.log('Error attempting to delete the RegistrationDefinition instance.')
            self.fail('Error deleting the RegistrationDefinition instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        try:
            response = self.mgmt_client.registration_definitions.get(scope=self.scope, registration_definition_id=self.registration_definition_id)
        except Exception as e:
            return False
        return response.as_dict()