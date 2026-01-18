import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestUnsetFWaaS(test_fakes.TestNeutronClientOSCV2):

    def test_unset_shared(self):
        target = self.resource['id']
        arglist = [target, '--share']
        verifylist = [(self.res, target), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': False})
        self.assertIsNone(result)

    def test_set_shared_and_no_shared(self):
        target = self.resource['id']
        arglist = [target, '--share', '--no-share']
        verifylist = [(self.res, target), ('share', True), ('no_share', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_duplicate_shared(self):
        target = self.resource['id']
        arglist = [target, '--share', '--share']
        verifylist = [(self.res, target), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': False})
        self.assertIsNone(result)