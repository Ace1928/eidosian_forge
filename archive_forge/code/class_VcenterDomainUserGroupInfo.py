from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VcenterDomainUserGroupInfo(PyVmomi):

    def __init__(self, module):
        super(VcenterDomainUserGroupInfo, self).__init__(module)
        self.domain = self.params['domain']
        self.search_string = self.params['search_string']
        self.belongs_to_group = self.params['belongs_to_group']
        self.belongs_to_user = self.params['belongs_to_user']
        self.exact_match = self.params['exact_match']
        self.find_users = self.params['find_users']
        self.find_groups = self.params['find_groups']

    def execute(self):
        user_directory_manager = self.content.userDirectory
        if not self.domain.upper() in user_directory_manager.domainList:
            self.module.fail_json(msg='domain not found: %s' % self.domain)
        try:
            user_search_result = user_directory_manager.RetrieveUserGroups(domain=self.domain, searchStr=self.search_string, belongsToGroup=self.belongs_to_group, belongsToUser=self.belongs_to_user, exactMatch=self.exact_match, findUsers=self.find_users, findGroups=self.find_groups)
        except vim.fault.NotFound as e:
            self.module.fail_json(msg='%s' % to_native(e.msg))
        except Exception as e:
            self.module.fail_json(msg="Couldn't gather domain user or group information: %s" % to_native(e))
        user_search_result_normalization = []
        if user_search_result:
            for object in user_search_result:
                user_search_result_normalization.append({'fullName': object.fullName, 'principal': object.principal, 'group': object.group})
        self.module.exit_json(changed=False, domain_user_groups=user_search_result_normalization)