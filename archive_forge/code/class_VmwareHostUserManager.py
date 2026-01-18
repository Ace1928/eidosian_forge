from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VmwareHostUserManager(PyVmomi):

    def __init__(self, module):
        super(VmwareHostUserManager, self).__init__(module)
        self.esxi_hostname = module.params['esxi_hostname']
        self.user_name = module.params['user_name']
        self.user_password = module.params['user_password']
        self.user_description = module.params['user_description']
        self.override_user_password = module.params['override_user_password']
        self.state = module.params['state']

    def search_user(self):
        """
        Search the specified user from ESXi

        Returns: searched user
        """
        searchStr = self.user_name
        exactMatch = True
        findUsers = True
        findGroups = False
        user_account = self.host_obj.configManager.userDirectory.RetrieveUserGroups(None, searchStr, None, None, exactMatch, findUsers, findGroups)
        return user_account

    def ensure_user_info_diff(self, user_account):
        """
        Ensure a user information difference.
        The method can check a user description difference only.
        Also, it can't get the set password via vSphere API.

        Returns: bool
        """
        if user_account.fullName != self.user_description and self.user_description is not None:
            return True
        return False

    def add_user(self):
        """
        Add a new user
        """
        user_spec = vim.host.LocalAccountManager.AccountSpecification(id=self.user_name, description=self.user_description, password=self.user_password)
        try:
            self.host_obj.configManager.accountManager.CreateUser(user_spec)
        except Exception as e:
            self.module.fail_json(msg='Failed to add a new user: %s' % to_text(e.msg))

    def update_user(self):
        """
        Update a user information
        """
        user_spec = vim.host.LocalAccountManager.AccountSpecification(id=self.user_name, description=self.user_description)
        if self.user_password and self.override_user_password:
            user_spec.password = self.user_password
        try:
            self.host_obj.configManager.accountManager.UpdateUser(user_spec)
        except Exception as e:
            self.module.fail_json(msg='Failed to update a new password: %s' % to_text(e))

    def remove_user(self):
        """
        Remove a user
        """
        try:
            self.host_obj.configManager.accountManager.RemoveUser(self.user_name)
        except Exception as e:
            self.module.fail_json(msg='Failed to remove a user: %s' % to_text(e.msg))

    def execute(self):
        self.host_obj = self.find_hostsystem_by_name(self.esxi_hostname)
        if self.host_obj is None:
            self.module.fail_json(msg='Cannot find the specified ESXi host: %s' % self.params['esxi_hostname'])
        user_account = self.search_user()
        changed = False
        msg = 'The change will not occur for the user information.'
        if self.state == 'present':
            if user_account:
                user_diff = self.ensure_user_info_diff(user_account[0])
                if user_diff or self.override_user_password:
                    changed = True
                    if self.module.check_mode:
                        msg = 'The user information will be updated.'
                    else:
                        msg = 'Updated the user information.'
                        self.update_user()
            else:
                changed = True
                if self.module.check_mode:
                    msg = 'The new user will be added.'
                else:
                    msg = 'Added the new user.'
                    self.add_user()
        if self.state == 'absent':
            if user_account:
                changed = True
                if self.module.check_mode:
                    msg = 'The user will be removed.'
                else:
                    msg = 'Removed the user.'
                    self.remove_user()
        self.module.exit_json(changed=changed, msg=msg)