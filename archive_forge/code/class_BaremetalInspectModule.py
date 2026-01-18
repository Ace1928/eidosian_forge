from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalInspectModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['uuid', 'id']), mac=dict())
    module_kwargs = dict(mutually_exclusive=[('name', 'mac')], required_one_of=[('name', 'mac')])

    def run(self):
        node_name_or_id = self.params['name']
        node = None
        if node_name_or_id is not None:
            node = self.conn.baremetal.find_node(node_name_or_id)
        else:
            node = self.conn.get_machine_by_mac(self.params['mac'])
        if node is None:
            self.fail_json(msg='node not found.')
        node = self.conn.inspect_machine(node['id'], wait=self.params['wait'], timeout=self.params['timeout'])
        node = node.to_dict(computed=False)
        self.exit_json(changed=True, node=node)