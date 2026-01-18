import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def _create_available_node(self):
    node = self.create_node(resource_class=self.resource_class)
    self.conn.baremetal.set_node_provision_state(node, 'manage', wait=True)
    self.conn.baremetal.set_node_provision_state(node, 'provide', wait=True)
    self.conn.baremetal.set_node_power_state(node, 'power off')
    self.addCleanup(lambda: self.conn.baremetal.update_node(node.id, instance_id=None))
    return node