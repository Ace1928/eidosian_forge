from __future__ import absolute_import, division, print_function
class AzureRMRoleAssignmentInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(assignee=dict(type='str', aliases=['assignee_object_id']), id=dict(type='str'), name=dict(type='str'), role_definition_id=dict(type='str'), scope=dict(type='str'), strict_scope_match=dict(type='bool', default=False))
        self.assignee = None
        self.id = None
        self.name = None
        self.role_definition_id = None
        self.scope = None
        self.strict_scope_match = None
        self.results = dict(changed=False, roleassignments=[])
        mutually_exclusive = [['name', 'assignee', 'id']]
        super(AzureRMRoleAssignmentInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True, mutually_exclusive=mutually_exclusive)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        is_old_facts = self.module._name == 'azure_rm_roleassignment_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_roleassignment_facts' module has been renamed to 'azure_rm_roleassignment_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.id:
            self.results['roleassignments'] = self.get_by_id()
        elif self.name and self.scope:
            self.results['roleassignments'] = self.get_by_name()
        elif self.name and (not self.scope):
            self.fail('Parameter Error: Name requires a scope to also be set.')
        elif self.scope:
            self.results['roleassignments'] = self.list_by_scope()
        elif self.assignee:
            self.results['roleassignments'] = self.list_by_assignee()
        else:
            self.results['roleassignments'] = self.list_assignments()
        return self.results

    def get_by_id(self):
        """
        Gets the role assignments by specific assignment id.

        :return: deserialized role assignment dictionary
        """
        self.log('Lists role assignment by id {0}'.format(self.id))
        results = []
        try:
            response = [self.authorization_client.role_assignments.get_by_id(role_id=self.id)]
            response = [self.roleassignment_to_dict(a) for a in response]
            results = response
        except Exception as ex:
            self.log("Didn't find role assignments id {0}".format(self.scope))
        return results

    def get_by_name(self):
        """
        Gets the properties of the specified role assignment by name.

        :return: deserialized role assignment dictionary
        """
        self.log('Gets role assignment {0} by name'.format(self.name))
        results = []
        try:
            response = [self.authorization_client.role_assignments.get(scope=self.scope, role_assignment_name=self.name)]
            response = [self.roleassignment_to_dict(a) for a in response]
            if self.role_definition_id:
                response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
            results = response
        except Exception as ex:
            self.log("Didn't find role assignment {0} in scope {1}".format(self.name, self.scope))
        return results

    def list_by_assignee(self):
        """
        Gets the role assignments by assignee.

        :return: deserialized role assignment dictionary
        """
        self.log('Gets role assignment {0} by name'.format(self.name))
        filter = "principalId eq '{0}'".format(self.assignee)
        return self.list_assignments(filter=filter)

    def list_assignments(self, filter=None):
        """
        Returns a list of assignments.
        """
        results = []
        try:
            response = list(self.authorization_client.role_assignments.list(filter=filter))
            response = [self.roleassignment_to_dict(a) for a in response]
            if self.role_definition_id:
                response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
            results = response
        except Exception as ex:
            self.log("Didn't find role assignments in subscription {0}.".format(self.subscription_id))
        return results

    def list_by_scope(self):
        """
        Lists the role assignments by specific scope.

        :return: deserialized role assignment dictionary
        """
        self.log('Lists role assignment by scope {0}'.format(self.scope))
        results = []
        try:
            response = list(self.authorization_client.role_assignments.list_for_scope(scope=self.scope, filter='atScope()'))
            response = [self.roleassignment_to_dict(role_assignment) for role_assignment in response]
            if self.assignee:
                response = [role_assignment for role_assignment in response if role_assignment.get('principal_id').lower() == self.assignee.lower()]
            if self.strict_scope_match:
                response = [role_assignment for role_assignment in response if role_assignment.get('scope').lower() == self.scope.lower()]
            if self.role_definition_id:
                response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
            results = response
        except Exception as ex:
            self.log("Didn't find role assignments at scope {0}".format(self.scope))
        return results

    def roleassignment_to_dict(self, assignment):
        return dict(assignee_object_id=assignment.principal_id, id=assignment.id, name=assignment.name, principal_id=assignment.principal_id, principal_type=assignment.principal_type, role_definition_id=assignment.role_definition_id, scope=assignment.scope, type=assignment.type)