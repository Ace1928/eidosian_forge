from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class PermissionsModule(BaseModule):

    def _user(self):
        user = search_by_attributes(self._connection.system_service().users_service(), usrname='{name}@{authz_name}'.format(name=self._module.params['user_name'], authz_name=self._module.params['authz_name']))
        if user is None:
            raise Exception("User '%s' was not found." % self._module.params['user_name'])
        return user

    def _group(self):
        groups = self._connection.system_service().groups_service().list(search='name="{name}"'.format(name=self._module.params['group_name']))
        if len(groups) > 1:
            groups = [g for g in groups if equal(self._module.params['namespace'], g.namespace) and equal(self._module.params['authz_name'], g.domain.name)]
        if not groups:
            raise Exception("Group '%s' was not found." % self._module.params['group_name'])
        return groups[0]

    def build_entity(self):
        entity = self._group() if self._module.params['group_name'] else self._user()
        return otypes.Permission(user=otypes.User(id=entity.id) if self._module.params['user_name'] else None, group=otypes.Group(id=entity.id) if self._module.params['group_name'] else None, role=otypes.Role(name=self._module.params['role']))