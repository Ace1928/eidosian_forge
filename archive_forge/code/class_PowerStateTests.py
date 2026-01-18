from ironicclient.tests.functional.osc.v1 import base
class PowerStateTests(base.TestCase):
    """Functional tests for baremetal node power state commands."""

    def setUp(self):
        super(PowerStateTests, self).setUp()
        self.node = self.node_create()

    def test_off_reboot_on(self):
        """Reboot node from Power OFF state.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set node Power State OFF as precondition.
        3) Call reboot command for baremetal node.
        4) Check node Power State ON in node properties.
        """
        self.openstack('baremetal node power off {0}'.format(self.node['uuid']))
        show_prop = self.node_show(self.node['uuid'], ['power_state'])
        self.assertEqual('power off', show_prop['power_state'])
        self.openstack('baremetal node reboot {0}'.format(self.node['uuid']))
        show_prop = self.node_show(self.node['uuid'], ['power_state'])
        self.assertEqual('power on', show_prop['power_state'])

    def test_on_reboot_on(self):
        """Reboot node from Power ON state.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set node Power State ON as precondition.
        3) Call reboot command for baremetal node.
        4) Check node Power State ON in node properties.
        """
        self.openstack('baremetal node power on {0}'.format(self.node['uuid']))
        show_prop = self.node_show(self.node['uuid'], ['power_state'])
        self.assertEqual('power on', show_prop['power_state'])
        self.openstack('baremetal node reboot {0}'.format(self.node['uuid']))
        show_prop = self.node_show(self.node['uuid'], ['power_state'])
        self.assertEqual('power on', show_prop['power_state'])