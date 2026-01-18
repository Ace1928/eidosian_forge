from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _update_security_group_rules(self, security_group, update):
    delete_security_group_rules = update.get('delete_security_group_rules')
    if delete_security_group_rules:
        for security_group_rule in delete_security_group_rules:
            self.conn.network.delete_security_group_rule(security_group_rule['id'])
    create_security_group_rules = update.get('create_security_group_rules')
    if create_security_group_rules:
        self.conn.network.create_security_group_rules(create_security_group_rules)
    if create_security_group_rules or delete_security_group_rules:
        return self.conn.network.get_security_group(security_group.id)
    else:
        return security_group