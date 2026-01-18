from __future__ import absolute_import, division, print_function
class AzureRMRoleAssignment(AzureRMModuleBase):
    """Configuration class for an Azure RM Role Assignment"""

    def __init__(self):
        self.module_arg_spec = dict(assignee_object_id=dict(type='str', aliases=['assignee']), id=dict(type='str'), name=dict(type='str'), role_definition_id=dict(type='str'), scope=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.assignee_object_id = None
        self.id = None
        self.name = None
        self.role_definition_id = None
        self.scope = None
        self.state = None
        self.results = dict(changed=False, id=None)
        mutually_exclusive = [['name', 'id'], ['scope', 'id']]
        required_one_of = [['scope', 'id']]
        required_if = [['state', 'present', ['assignee_object_id', 'role_definition_id']]]
        super(AzureRMRoleAssignment, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, required_one_of=required_one_of, required_if=required_if, mutually_exclusive=mutually_exclusive)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and (not self.scope):
            self.fail('Parameter Error: setting name requires a scope to also be set.')
        existing_assignment = None
        response = None
        existing_assignment = self.get_roleassignment()
        if existing_assignment:
            self.set_results(existing_assignment)
        if self.state == 'present':
            if not existing_assignment:
                self.log("Role assignment doesn't exist in this scope")
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                response = self.create_roleassignment()
                self.set_results(response)
            else:
                self.log('Role assignment already exists, not updatable')
                self.log('Result: {0}'.format(existing_assignment))
        elif self.state == 'absent':
            if existing_assignment:
                self.log('Delete role assignment')
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.delete_roleassignment(existing_assignment.get('id'))
                self.log('role assignment deleted')
            else:
                self.log('role assignment {0} does not exist.'.format(self.name))
        return self.results

    def create_roleassignment(self):
        """
        Creates role assignment.

        :return: deserialized role assignment
        """
        self.log('Creating role assignment {0}'.format(self.name))
        response = None
        try:
            parameters = self.authorization_models.RoleAssignmentCreateParameters(role_definition_id=self.role_definition_id, principal_id=self.assignee_object_id)
            if self.id:
                response = self.authorization_client.role_assignments.create_by_id(role_id=self.id, parameters=parameters)
            elif self.scope:
                if not self.name:
                    self.name = str(uuid.uuid4())
                response = self.authorization_client.role_assignments.create(scope=self.scope, role_assignment_name=self.name, parameters=parameters)
        except Exception as exc:
            self.log('Error attempting to create role assignment.')
            self.fail('Error creating role assignment: {0}'.format(str(exc)))
        return self.roleassignment_to_dict(response)

    def delete_roleassignment(self, assignment_id):
        """
        Deletes specified role assignment.

        :return: True
        """
        self.log('Deleting the role assignment {0}'.format(self.name))
        try:
            response = self.authorization_client.role_assignments.delete_by_id(role_id=assignment_id)
        except Exception as e:
            self.log('Error attempting to delete the role assignment.')
            self.fail('Error deleting the role assignment: {0}'.format(str(e)))
        return True

    def get_roleassignment(self):
        """
        Gets the properties of the specified role assignment.

        :return: deserialized role assignment dictionary
        """
        self.log('Checking if the role assignment {0} is present'.format(self.name))
        role_assignment = None
        if self.id:
            try:
                response = self.authorization_client.role_assignments.get_by_id(role_id=self.id)
                role_assignment = self.roleassignment_to_dict(response)
                if role_assignment and self.assignee_object_id and (role_assignment.get('assignee_object_id') != self.assignee_object_id):
                    self.fail('State Mismatch Error: The assignment ID exists, but does not match the provided assignee.')
                if role_assignment and self.role_definition_id and (role_assignment.get('role_definition_id').split('/')[-1].lower() != self.role_definition_id.split('/')[-1].lower()):
                    self.fail('State Mismatch Error: The assignment ID exists, but does not match the provided role.')
            except Exception as ex:
                self.log("Didn't find role assignments id {0}".format(self.id))
        elif self.name and self.scope:
            try:
                response = self.authorization_client.role_assignments.get(scope=self.scope, role_assignment_name=self.name)
                role_assignment = self.roleassignment_to_dict(response)
                if role_assignment and self.assignee_object_id and (role_assignment.get('assignee_object_id') != self.assignee_object_id):
                    self.fail('State Mismatch Error: The assignment name exists, but does not match the provided assignee.')
                if role_assignment and self.role_definition_id and (role_assignment.get('role_definition_id').split('/')[-1].lower() != self.role_definition_id.split('/')[-1].lower()):
                    self.fail('State Mismatch Error: The assignment name exists, but does not match the provided role.')
            except Exception as ex:
                self.log("Didn't find role assignment by name {0} at scope {1}".format(self.name, self.scope))
        else:
            try:
                if self.scope and self.assignee_object_id and self.role_definition_id:
                    response = list(self.authorization_client.role_assignments.list())
                    response = [self.roleassignment_to_dict(role_assignment) for role_assignment in response]
                    response = [role_assignment for role_assignment in response if role_assignment.get('scope') == self.scope]
                    response = [role_assignment for role_assignment in response if role_assignment.get('assignee_object_id') == self.assignee_object_id]
                    response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
                else:
                    self.fail('If id or name are not supplied, then assignee_object_id and role_definition_id are required.')
                if response:
                    role_assignment = response[0]
            except Exception as ex:
                self.log("Didn't find role assignments for subscription {0}".format(self.subscription_id))
        return role_assignment

    def set_results(self, assignment):
        self.results['id'] = assignment.get('id')
        self.results['name'] = assignment.get('name')
        self.results['type'] = assignment.get('type')
        self.results['assignee_object_id'] = assignment.get('assignee_object_id')
        self.results['principal_type'] = assignment.get('principal_type')
        self.results['role_definition_id'] = assignment.get('role_definition_id')
        self.results['scope'] = assignment.get('scope')

    def roleassignment_to_dict(self, assignment):
        return dict(assignee_object_id=assignment.principal_id, id=assignment.id, name=assignment.name, principal_id=assignment.principal_id, principal_type=assignment.principal_type, role_definition_id=assignment.role_definition_id, scope=assignment.scope, type=assignment.type)