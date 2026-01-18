import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.ddt
class BaremetalNodeNegativeTests(base.TestCase):
    """Negative tests for baremetal node commands."""

    def setUp(self):
        super(BaremetalNodeNegativeTests, self).setUp()
        self.node = self.node_create()

    @ddt.data(('', '', 'error: the following arguments are required: --driver'), ('--driver', 'wrongdriver', 'No valid host was found. Reason: No conductor service registered which supports driver wrongdriver.'))
    @ddt.unpack
    def test_create_driver(self, argument, value, ex_text):
        """Negative test for baremetal node driver options."""
        base_cmd = 'baremetal node create'
        command = self.construct_cmd(base_cmd, argument, value)
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    def test_delete_no_node(self):
        """Test for baremetal node delete without node specified."""
        command = 'baremetal node delete'
        ex_text = ''
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    def test_list_wrong_argument(self):
        """Test for baremetal node list with wrong argument."""
        command = 'baremetal node list --wrong_arg'
        ex_text = 'error: unrecognized arguments: --wrong_arg'
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    @ddt.data(('--property', '', 'error: the following arguments are required: <node>'), ('--property', 'prop', 'Attributes must be a list of PATH=VALUE'))
    @ddt.unpack
    def test_set_property(self, argument, value, ex_text):
        """Negative test for baremetal node set command options."""
        base_cmd = 'baremetal node set'
        command = self.construct_cmd(base_cmd, argument, value, self.node['uuid'])
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    @ddt.data(('--property', '', 'error: the following arguments are required: <node>'), ('--property', 'prop', "Reason: can't remove a non-existent object"))
    @ddt.unpack
    def test_unset_property(self, argument, value, ex_text):
        """Negative test for baremetal node unset command options."""
        base_cmd = 'baremetal node unset'
        command = self.construct_cmd(base_cmd, argument, value, self.node['uuid'])
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)