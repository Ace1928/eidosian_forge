from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalNodeInfoModule(OpenStackModule):
    argument_spec = dict(mac=dict(), name=dict(aliases=['node']))
    module_kwargs = dict(mutually_exclusive=[('mac', 'name')], supports_check_mode=True)

    def run(self):
        name_or_id = self.params['name']
        mac = self.params['mac']
        node_id = None
        if name_or_id:
            node = self.conn.baremetal.find_node(name_or_id)
            if node:
                node_id = node['id']
        elif mac:
            baremetal_port = self.conn.get_nic_by_mac(mac)
            if baremetal_port:
                node_id = baremetal_port['node_id']
        if name_or_id or mac:
            if node_id:
                node = self.conn.baremetal.get_node(node_id)
                nodes = [node.to_dict(computed=False)]
            else:
                nodes = []
        else:
            nodes = [node.to_dict(computed=False) for node in self.conn.baremetal.nodes(details=True)]
        self.exit_json(changed=False, nodes=nodes, baremetal_nodes=nodes)