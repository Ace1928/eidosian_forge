from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class PortInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['port']), filters=dict(type='dict'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        ports = [p.to_dict(computed=False) for p in self.conn.search_ports(name_or_id=self.params['name'], filters=self.params['filters'])]
        self.exit_json(changed=False, ports=ports)