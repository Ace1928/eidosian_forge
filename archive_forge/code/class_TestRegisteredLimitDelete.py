import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegisteredLimitDelete(TestRegisteredLimit):

    def setUp(self):
        super(TestRegisteredLimitDelete, self).setUp()
        self.cmd = registered_limit.DeleteRegisteredLimit(self.app, None)

    def test_registered_limit_delete(self):
        self.registered_limit_mock.delete.return_value = None
        arglist = [identity_fakes.registered_limit_id]
        verifylist = [('registered_limit_id', [identity_fakes.registered_limit_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.registered_limit_mock.delete.assert_called_with(identity_fakes.registered_limit_id)
        self.assertIsNone(result)

    def test_registered_limit_delete_with_exception(self):
        return_value = ksa_exceptions.NotFound()
        self.registered_limit_mock.delete.side_effect = return_value
        arglist = ['fake-registered-limit-id']
        verifylist = [('registered_limit_id', ['fake-registered-limit-id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 1 registered limits failed to delete.', str(e))