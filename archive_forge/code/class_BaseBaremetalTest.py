from openstack.tests.functional import base
class BaseBaremetalTest(base.BaseFunctionalTest):
    min_microversion = None
    node_id = None

    def setUp(self):
        super(BaseBaremetalTest, self).setUp()
        self.require_service('baremetal', min_microversion=self.min_microversion)

    def create_allocation(self, **kwargs):
        allocation = self.conn.baremetal.create_allocation(**kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_allocation(allocation.id, ignore_missing=True))
        return allocation

    def create_chassis(self, **kwargs):
        chassis = self.conn.baremetal.create_chassis(**kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_chassis(chassis.id, ignore_missing=True))
        return chassis

    def create_node(self, driver='fake-hardware', **kwargs):
        node = self.conn.baremetal.create_node(driver=driver, **kwargs)
        self.node_id = node.id
        self.addCleanup(lambda: self.conn.baremetal.delete_node(self.node_id, ignore_missing=True))
        self.assertIsNotNone(self.node_id)
        return node

    def create_port(self, node_id=None, **kwargs):
        node_id = node_id or self.node_id
        port = self.conn.baremetal.create_port(node_uuid=node_id, **kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_port(port.id, ignore_missing=True))
        return port

    def create_port_group(self, node_id=None, **kwargs):
        node_id = node_id or self.node_id
        port_group = self.conn.baremetal.create_port_group(node_uuid=node_id, **kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_port_group(port_group.id, ignore_missing=True))
        return port_group

    def create_volume_connector(self, node_id=None, **kwargs):
        node_id = node_id or self.node_id
        volume_connector = self.conn.baremetal.create_volume_connector(node_uuid=node_id, **kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_volume_connector(volume_connector.id, ignore_missing=True))
        return volume_connector

    def create_volume_target(self, node_id=None, **kwargs):
        node_id = node_id or self.node_id
        volume_target = self.conn.baremetal.create_volume_target(node_uuid=node_id, **kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_volume_target(volume_target.id, ignore_missing=True))
        return volume_target

    def create_deploy_template(self, **kwargs):
        """Create a new deploy_template from attributes."""
        deploy_template = self.conn.baremetal.create_deploy_template(**kwargs)
        self.addCleanup(lambda: self.conn.baremetal.delete_deploy_template(deploy_template.id, ignore_missing=True))
        return deploy_template