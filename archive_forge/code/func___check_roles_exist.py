from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def __check_roles_exist(self):
    if self.groups:
        existent_groups = self.__roles_exist(self.groups)
        for group in self.groups:
            if group not in existent_groups:
                if self.fail_on_role:
                    self.module.fail_json(msg='Role %s does not exist' % group)
                else:
                    self.module.warn('Role %s does not exist, pass' % group)
                    self.non_existent_roles.append(group)
    existent_roles = self.__roles_exist(self.target_roles)
    for role in self.target_roles:
        if role not in existent_roles:
            if self.fail_on_role:
                self.module.fail_json(msg='Role %s does not exist' % role)
            else:
                self.module.warn('Role %s does not exist, pass' % role)
            if role not in self.groups:
                self.non_existent_roles.append(role)
            elif self.fail_on_role:
                self.module.exit_json(msg="Role role '%s' is a member of role '%s'" % (role, role))
            else:
                self.module.warn("Role role '%s' is a member of role '%s', pass" % (role, role))
    if self.groups:
        self.groups = [g for g in self.groups if g not in self.non_existent_roles]
    self.target_roles = [r for r in self.target_roles if r not in self.non_existent_roles]