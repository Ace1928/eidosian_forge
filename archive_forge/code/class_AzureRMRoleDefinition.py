from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
class AzureRMRoleDefinition(AzureRMModuleBase):
    """Configuration class for an Azure RM Role definition resource"""

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), scope=dict(type='str'), permissions=dict(type='list', elements='dict', options=permission_spec), assignable_scopes=dict(type='list', elements='str'), description=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.name = None
        self.scope = None
        self.permissions = None
        self.description = None
        self.assignable_scopes = None
        self.results = dict(changed=False, id=None)
        self.state = None
        self.to_do = Actions.NoAction
        self.role = None
        self._client = None
        super(AzureRMRoleDefinition, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        old_response = None
        response = None
        self._client = self.get_mgmt_svc_client(AuthorizationManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2018-01-01-preview')
        self.scope = self.build_scope()
        old_response = self.get_roledefinition()
        if old_response:
            self.results['id'] = old_response['id']
            self.role = old_response
        if self.state == 'present':
            if not old_response:
                self.log("Role definition doesn't exist in this scope")
                self.to_do = Actions.CreateOrUpdate
            else:
                self.log('Role definition already exists')
                self.log('Result: {0}'.format(old_response))
                if self.check_update(old_response):
                    self.to_do = Actions.CreateOrUpdate
        elif self.state == 'absent':
            if old_response:
                self.log('Delete role definition')
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.delete_roledefinition(old_response['name'])
                self.log('role definition deleted')
            else:
                self.log('role definition {0} not exists.'.format(self.name))
        if self.to_do == Actions.CreateOrUpdate:
            self.log('Need to Create/Update role definition')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_roledefinition()
            self.results['id'] = response['id']
        return self.results

    def build_scope(self):
        subscription_scope = '/subscriptions/' + self.subscription_id
        if self.scope is None:
            return subscription_scope
        return self.scope

    def check_update(self, old_definition):
        if self.description and self.description != old_definition['description']:
            return True
        if self.permissions:
            if len(self.permissions) != len(old_definition['permissions']):
                return True
            existing_permissions = self.permissions_to_set(old_definition['permissions'])
            new_permissions = self.permissions_to_set(self.permissions)
            if existing_permissions != new_permissions:
                return True
        if self.assignable_scopes and self.assignable_scopes != old_definition['assignable_scopes']:
            return True
        return False

    def permissions_to_set(self, permissions):
        new_permissions = [str(dict(actions=set([to_native(a) for a in item.get('actions')]) if item.get('actions') else None, not_actions=set([to_native(a) for a in item.get('not_actions')]) if item.get('not_actions') else None, data_actions=set([to_native(a) for a in item.get('data_actions')]) if item.get('data_actions') else None, not_data_actions=set([to_native(a) for a in item.get('not_data_actions')]) if item.get('not_data_actions') else None)) for item in permissions]
        return set(new_permissions)

    def create_update_roledefinition(self):
        """
        Creates or updates role definition.

        :return: deserialized role definition
        """
        self.log('Creating / Updating role definition {0}'.format(self.name))
        try:
            permissions = None
            if self.permissions:
                permissions = [AuthorizationManagementClient.models('2018-01-01-preview').Permission(actions=p.get('actions', None), not_actions=p.get('not_actions', None), data_actions=p.get('data_actions', None), not_data_actions=p.get('not_data_actions', None)) for p in self.permissions]
            role_definition = AuthorizationManagementClient.models('2018-01-01-preview').RoleDefinition(role_name=self.name, description=self.description, permissions=permissions, assignable_scopes=self.assignable_scopes, role_type='CustomRole')
            if self.role:
                role_definition.name = self.role['name']
            response = self._client.role_definitions.create_or_update(role_definition_id=self.role['name'] if self.role else str(uuid.uuid4()), scope=self.scope, role_definition=role_definition)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create role definition.')
            self.fail('Error creating role definition: {0}'.format(str(exc)))
        return roledefinition_to_dict(response)

    def delete_roledefinition(self, role_definition_id):
        """
        Deletes specified role definition.

        :return: True
        """
        self.log('Deleting the role definition {0}'.format(self.name))
        scope = self.build_scope()
        try:
            response = self._client.role_definitions.delete(scope=scope, role_definition_id=role_definition_id)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to delete the role definition.')
            self.fail('Error deleting the role definition: {0}'.format(str(e)))
        return True

    def get_roledefinition(self):
        """
        Gets the properties of the specified role definition.

        :return: deserialized role definition state dictionary
        """
        self.log('Checking if the role definition {0} is present'.format(self.name))
        response = None
        try:
            response = list(self._client.role_definitions.list(scope=self.scope))
            if len(response) > 0:
                self.log('Response : {0}'.format(response))
                roles = []
                for r in response:
                    if r.role_name == self.name:
                        roles.append(r)
                if len(roles) == 1:
                    self.log('role definition : {0} found'.format(self.name))
                    return roledefinition_to_dict(roles[0])
                if len(roles) > 1:
                    self.fail('Found multiple role definitions: {0}'.format(roles))
        except Exception as ex:
            self.log("Didn't find role definition {0}".format(self.name))
        return False