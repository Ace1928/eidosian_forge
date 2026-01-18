import testtools
from osc_lib import exceptions
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestSetVPNaaS(test_fakes.TestNeutronClientOSCV2):

    def test_set_name(self):
        target = self.resource['id']
        update = 'change'
        arglist = [target, '--name', update]
        verifylist = [(self.res, target), ('name', update)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'name': update})
        self.assertIsNone(result)

    def test_set_description(self):
        target = self.resource['id']
        update = 'change-desc'
        arglist = [target, '--description', update]
        verifylist = [(self.res, target), ('description', update)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'description': update})
        self.assertIsNone(result)