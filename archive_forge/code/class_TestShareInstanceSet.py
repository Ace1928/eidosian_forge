from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceSet(TestShareInstance):

    def setUp(self):
        super(TestShareInstanceSet, self).setUp()
        self.share_instance = manila_fakes.FakeShareInstance.create_one_share_instance()
        self.instances_mock.get.return_value = self.share_instance
        self.cmd = osc_share_instances.ShareInstanceSet(self.app, None)

    def test_share_instance_set_status(self):
        new_status = 'available'
        arglist = [self.share_instance.id, '--status', new_status]
        verifylist = [('instance', self.share_instance.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.instances_mock.reset_state.assert_called_with(self.share_instance, new_status)
        self.assertIsNone(result)

    def test_share_instance_set_status_exception(self):
        new_status = 'available'
        arglist = [self.share_instance.id, '--status', new_status]
        verifylist = [('instance', self.share_instance.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.instances_mock.reset_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_instance_set_nothing_defined(self):
        arglist = [self.share_instance.id]
        verifylist = [('instance', self.share_instance.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)