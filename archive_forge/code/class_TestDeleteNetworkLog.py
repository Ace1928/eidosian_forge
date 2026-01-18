import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
class TestDeleteNetworkLog(TestNetworkLog):

    def setUp(self):
        super(TestDeleteNetworkLog, self).setUp()
        self.neutronclient.delete_network_log = mock.Mock(return_value={'log': self.res})
        self.mocked = self.neutronclient.delete_network_log
        self.cmd = network_log.DeleteNetworkLog(self.app, self.namespace)

    def test_delete_with_one_resource(self):
        target = self.res['id']
        arglist = [target]
        verifylist = [('network_log', [target])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target)
        self.assertIsNone(result)

    def test_delete_with_multiple_resources(self):
        target1 = 'target1'
        target2 = 'target2'
        arglist = [target1, target2]
        verifylist = [('network_log', [target1, target2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.assertEqual(2, self.mocked.call_count)
        for idx, reference in enumerate([target1, target2]):
            actual = ''.join(self.mocked.call_args_list[idx][0])
            self.assertEqual(reference, actual)

    def test_delete_with_no_exist_id(self):
        self.neutronclient.find_resource.side_effect = Exception
        target = 'not_exist'
        arglist = [target]
        verifylist = [('network_log', [target])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)