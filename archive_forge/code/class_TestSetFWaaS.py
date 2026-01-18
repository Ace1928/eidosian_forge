import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestSetFWaaS(test_fakes.TestNeutronClientOSCV2):

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

    def test_set_shared(self):
        target = self.resource['id']
        arglist = [target, '--share']
        verifylist = [(self.res, target), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': True})
        self.assertIsNone(result)

    def test_set_duplicate_shared(self):
        target = self.resource['id']
        arglist = [target, '--share', '--share']
        verifylist = [(self.res, target), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': True})
        self.assertIsNone(result)

    def test_set_no_share(self):
        target = self.resource['id']
        arglist = [target, '--no-share']
        verifylist = [(self.res, target), ('share', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': False})
        self.assertIsNone(result)

    def test_set_duplicate_no_share(self):
        target = self.resource['id']
        arglist = [target, '--no-share', '--no-share']
        verifylist = [(self.res, target), ('no_share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'shared': False})
        self.assertIsNone(result)

    def test_set_no_share_and_shared(self):
        target = self.resource['id']
        arglist = [target, '--no-share', '--share']
        verifylist = [(self.res, target), ('no_share', True), ('share', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_shared_and_no_share(self):
        target = self.resource['id']
        arglist = [target, '--share', '--no_share']
        verifylist = [(self.res, target), ('share', True), ('no_share', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_project(self):
        target = self.resource['id']
        project_id = 'b14ce3b699594d13819a859480286489'
        arglist = [target, '--project', project_id]
        verifylist = [(self.res, target), ('tenant_id', project_id)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_project_domain(self):
        target = self.resource['id']
        project_domain = 'mydomain.com'
        arglist = [target, '--project-domain', project_domain]
        verifylist = [(self.res, target), ('project_domain', project_domain)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)