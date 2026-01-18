import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.ddt
class BaremetalNodeCreateNegativeTests(base.TestCase):
    """Negative tests for node create command."""

    def setUp(self):
        super(BaremetalNodeCreateNegativeTests, self).setUp()

    @ddt.data(('--uuid', '', 'expected one argument'), ('--uuid', '!@#$^*&%^', 'Expected UUID for uuid'), ('--uuid', '0000 0000', 'unrecognized arguments'), ('--driver-info', '', 'expected one argument'), ('--driver-info', 'some info', 'unrecognized arguments'), ('--property', '', 'expected one argument'), ('--property', 'some property', 'unrecognized arguments'), ('--extra', '', 'expected one argument'), ('--extra', 'some extra', 'unrecognized arguments'), ('--name', '', 'expected one argument'), ('--name', 'some name', 'unrecognized arguments'), ('--network-interface', '', 'expected one argument'), ('--resource-class', '', 'expected one argument'))
    @ddt.unpack
    def test_baremetal_node_create(self, argument, value, ex_text):
        base_cmd = 'baremetal node create --driver %s' % self.driver_name
        command = self.construct_cmd(base_cmd, argument, value)
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)