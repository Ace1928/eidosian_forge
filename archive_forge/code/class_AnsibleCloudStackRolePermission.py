from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackRolePermission(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackRolePermission, self).__init__(module)
        cloudstack_min_version = LooseVersion('4.9.2')
        self.returns = {'id': 'id', 'roleid': 'role_id', 'rule': 'name', 'permission': 'permission', 'description': 'description'}
        self.role_permission = None
        self.cloudstack_version = self._cloudstack_ver()
        if self.cloudstack_version < cloudstack_min_version:
            self.fail_json(msg='This module requires CloudStack >= %s.' % cloudstack_min_version)

    def _cloudstack_ver(self):
        capabilities = self.get_capabilities()
        return LooseVersion(capabilities['cloudstackversion'])

    def _get_role_id(self):
        role = self.module.params.get('role')
        if not role:
            return None
        res = self.query_api('listRoles')
        roles = res['role']
        if roles:
            for r in roles:
                if role in [r['name'], r['id']]:
                    return r['id']
        self.fail_json(msg="Role '%s' not found" % role)

    def _get_role_perm(self):
        role_permission = self.role_permission
        args = {'roleid': self._get_role_id()}
        rp = self.query_api('listRolePermissions', **args)
        if rp:
            role_permission = rp['rolepermission']
        return role_permission

    def _get_rule(self, rule=None):
        if not rule:
            rule = self.module.params.get('name')
        if self._get_role_perm():
            for _rule in self._get_role_perm():
                if rule == _rule['rule'] or rule == _rule['id']:
                    return _rule
        return None

    def _get_rule_order(self):
        perms = self._get_role_perm()
        rules = []
        if perms:
            for i, rule in enumerate(perms):
                rules.append(rule['id'])
        return rules

    def replace_rule(self):
        old_rule = self._get_rule()
        if old_rule:
            rules_order = self._get_rule_order()
            old_pos = rules_order.index(old_rule['id'])
            self.remove_role_perm()
            new_rule = self.create_role_perm()
            if new_rule:
                perm_order = self.order_permissions(int(old_pos - 1), new_rule['id'])
                return perm_order
        return None

    def order_permissions(self, parent, rule_id):
        rules = self._get_rule_order()
        if isinstance(parent, int):
            parent_pos = parent
        elif parent == '0':
            parent_pos = -1
        else:
            parent_rule = self._get_rule(parent)
            if not parent_rule:
                self.fail_json(msg="Parent rule '%s' not found" % parent)
            parent_pos = rules.index(parent_rule['id'])
        r_id = rules.pop(rules.index(rule_id))
        rules.insert(parent_pos + 1, r_id)
        rules = ','.join(map(str, rules))
        return rules

    def create_or_update_role_perm(self):
        role_permission = self._get_rule()
        if not role_permission:
            role_permission = self.create_role_perm()
        else:
            role_permission = self.update_role_perm(role_permission)
        return role_permission

    def create_role_perm(self):
        role_permission = None
        self.result['changed'] = True
        args = {'rule': self.module.params.get('name'), 'description': self.module.params.get('description'), 'roleid': self._get_role_id(), 'permission': self.module.params.get('permission')}
        if not self.module.check_mode:
            res = self.query_api('createRolePermission', **args)
            role_permission = res['rolepermission']
        return role_permission

    def update_role_perm(self, role_perm):
        perm_order = None
        if not self.module.params.get('parent'):
            args = {'ruleid': role_perm['id'], 'roleid': role_perm['roleid'], 'permission': self.module.params.get('permission')}
            if self.has_changed(args, role_perm, only_keys=['permission']):
                self.result['changed'] = True
                if not self.module.check_mode:
                    if self.cloudstack_version >= LooseVersion('4.11.0'):
                        self.query_api('updateRolePermission', **args)
                        role_perm = self._get_rule()
                    else:
                        perm_order = self.replace_rule()
        else:
            perm_order = self.order_permissions(self.module.params.get('parent'), role_perm['id'])
        if perm_order:
            args = {'roleid': role_perm['roleid'], 'ruleorder': perm_order}
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('updateRolePermission', **args)
                role_perm = self._get_rule()
        return role_perm

    def remove_role_perm(self):
        role_permission = self._get_rule()
        if role_permission:
            self.result['changed'] = True
            args = {'id': role_permission['id']}
            if not self.module.check_mode:
                self.query_api('deleteRolePermission', **args)
        return role_permission