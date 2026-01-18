from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class SecurityGroupRuleInfoModule(OpenStackModule):
    argument_spec = dict(description=dict(), direction=dict(choices=['egress', 'ingress']), ether_type=dict(choices=['IPv4', 'IPv6'], aliases=['ethertype']), id=dict(aliases=['rule']), port_range_min=dict(type='int'), port_range_max=dict(type='int'), project=dict(), protocol=dict(), remote_group=dict(), remote_ip_prefix=dict(), revision_number=dict(type='int'), security_group=dict())
    module_kwargs = dict(mutually_exclusive=[('remote_ip_prefix', 'remote_group')], supports_check_mode=True)

    def run(self):
        filters = dict(((k, self.params[k]) for k in ['description', 'direction', 'ether_type', 'id', 'port_range_min', 'port_range_max', 'protocol', 'remote_group', 'revision_number', 'remote_ip_prefix'] if self.params[k] is not None))
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.find_project(project_name_or_id)
            if not project:
                self.exit_json(changed=False, security_group_rules=[])
            filters['project_id'] = project.id
        security_group_name_or_id = self.params['security_group']
        if security_group_name_or_id is not None:
            security_group = self.conn.network.find_security_group(security_group_name_or_id)
            if not security_group:
                self.exit_json(changed=False, security_group_rules=[])
            filters['security_group_id'] = security_group.id
        security_group_rules = self.conn.network.security_group_rules(**filters)
        self.exit_json(changed=False, security_group_rules=[r.to_dict(computed=False) for r in security_group_rules])