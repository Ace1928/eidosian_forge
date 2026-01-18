from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VMwareLocalRoleManager(PyVmomi):
    """Class to manage local roles"""

    def __init__(self, module):
        super(VMwareLocalRoleManager, self).__init__(module)
        self.module = module
        self.params = module.params
        self.role_name = self.params['local_role_name']
        self.state = self.params['state']
        self.priv_ids = self.params['local_privilege_ids']
        self.force = not self.params['force_remove']
        self.current_role = None
        self.action = self.params['action']
        if self.content.authorizationManager is None:
            self.module.fail_json(msg='Failed to get local authorization manager settings.', details="It seems that '%s' does not support this functionality" % self.params['hostname'])

    def process_state(self):
        """Process the state of the local role"""
        local_role_manager_states = {'absent': {'present': self.state_remove_role, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_update_role, 'absent': self.state_create_role}}
        try:
            local_role_manager_states[self.state][self.check_local_role_manager_state()]()
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=runtime_fault.msg)
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=method_fault.msg)
        except Exception as e:
            self.module.fail_json(msg=str(e))

    def check_local_role_manager_state(self):
        """Check local roles"""
        auth_role = self.find_authorization_role()
        if auth_role:
            self.current_role = auth_role
            return 'present'
        return 'absent'

    def find_authorization_role(self):
        """Find local role"""
        desired_role = None
        for role in self.content.authorizationManager.roleList:
            if role.name == self.role_name:
                desired_role = role
        return desired_role

    def state_create_role(self):
        """Create local role"""
        role_id = None
        results = dict()
        results['role_name'] = self.role_name
        results['privileges'] = self.priv_ids
        results['local_role_name'] = self.role_name
        results['new_privileges'] = self.priv_ids
        if self.module.check_mode:
            results['msg'] = 'Role would be created'
        else:
            try:
                role_id = self.content.authorizationManager.AddAuthorizationRole(name=self.role_name, privIds=self.priv_ids)
                results['role_id'] = role_id
                results['msg'] = 'Role created'
            except vim.fault.AlreadyExists as already_exists:
                self.module.fail_json(msg="Failed to create role '%s' as the user specified role name already exists." % self.role_name, details=already_exists.msg)
            except vim.fault.InvalidName as invalid_name:
                self.module.fail_json(msg='Failed to create a role %s as the user specified role name is empty' % self.role_name, details=invalid_name.msg)
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg='Failed to create a role %s as the user specified privileges are unknown' % self.role_name, etails=invalid_argument.msg)
        self.module.exit_json(changed=True, result=results)

    def state_remove_role(self):
        """Remove local role"""
        results = dict()
        results['role_name'] = self.role_name
        results['role_id'] = self.current_role.roleId
        results['local_role_name'] = self.role_name
        if self.module.check_mode:
            results['msg'] = 'Role would be deleted'
        else:
            try:
                self.content.authorizationManager.RemoveAuthorizationRole(roleId=self.current_role.roleId, failIfUsed=self.force)
                results['msg'] = 'Role deleted'
            except vim.fault.NotFound as not_found:
                self.module.fail_json(msg='Failed to remove a role %s as the user specified role name does not exist.' % self.role_name, details=not_found.msg)
            except vim.fault.RemoveFailed as remove_failed:
                msg = "Failed to remove role '%s' as the user specified role name." % self.role_name
                if self.force:
                    msg += ' Use force_remove as True.'
                self.module.fail_json(msg=msg, details=remove_failed.msg)
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg='Failed to remove a role %s as the user specified role is a system role' % self.role_name, details=invalid_argument.msg)
        self.module.exit_json(changed=True, result=results)

    def state_exit_unchanged(self):
        """Don't do anything"""
        results = dict()
        results['role_name'] = self.role_name
        results['local_role_name'] = self.role_name
        results['msg'] = 'Role not present'
        self.module.exit_json(changed=False, result=results)

    def state_update_role(self):
        """Update local role"""
        changed = False
        changed_privileges = []
        results = dict()
        results['role_name'] = self.role_name
        results['role_id'] = self.current_role.roleId
        results['local_role_name'] = self.role_name
        current_privileges = self.current_role.privilege
        results['privileges'] = current_privileges
        results['new_privileges'] = current_privileges
        if self.action == 'add':
            for priv in self.params['local_privilege_ids']:
                if priv not in current_privileges:
                    changed_privileges.append(priv)
                    changed = True
            if changed:
                changed_privileges.extend(current_privileges)
        elif self.action == 'set':
            self.params['local_privilege_ids'].extend(['System.Anonymous', 'System.Read', 'System.View'])
            changed_privileges = self.params['local_privilege_ids']
            changes_applied = list(set(current_privileges) ^ set(changed_privileges))
            if changes_applied:
                changed = True
        elif self.action == 'remove':
            changed_privileges = list(current_privileges)
            for priv in self.params['local_privilege_ids']:
                if priv in current_privileges:
                    changed = True
                    changed_privileges.remove(priv)
        if changed:
            results['privileges'] = changed_privileges
            results['privileges_previous'] = current_privileges
            results['new_privileges'] = changed_privileges
            results['old_privileges'] = current_privileges
            if self.module.check_mode:
                results['msg'] = 'Role privileges would be updated'
            else:
                try:
                    self.content.authorizationManager.UpdateAuthorizationRole(roleId=self.current_role.roleId, newName=self.current_role.name, privIds=changed_privileges)
                    results['msg'] = 'Role privileges updated'
                except vim.fault.NotFound as not_found:
                    self.module.fail_json(msg='Failed to update role. Please check privileges provided for update', details=not_found.msg)
                except vim.fault.InvalidName as invalid_name:
                    self.module.fail_json(msg='Failed to update role as role name is empty', details=invalid_name.msg)
                except vim.fault.AlreadyExists as already_exists:
                    self.module.fail_json(msg='Failed to update role', details=already_exists.msg)
                except vmodl.fault.InvalidArgument as invalid_argument:
                    self.module.fail_json(msg='Failed to update role as user specified role is system role which can not be changed', details=invalid_argument.msg)
                except vim.fault.NoPermission as no_permission:
                    self.module.fail_json(msg="Failed to update role as current session doesn't have any privilege to update specified role", details=no_permission.msg)
        else:
            results['msg'] = 'Role privileges are properly configured'
        self.module.exit_json(changed=changed, result=results)