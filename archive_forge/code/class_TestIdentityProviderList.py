import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
class TestIdentityProviderList(TestIdentityProvider):

    def setUp(self):
        super(TestIdentityProviderList, self).setUp()
        self.identity_providers_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
        self.identity_providers_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)]
        self.cmd = identity_provider.ListIdentityProvider(self.app, None)

    def test_identity_provider_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.identity_providers_mock.list.assert_called_with()
        collist = ('ID', 'Enabled', 'Domain ID', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.idp_id, True, identity_fakes.domain_id, identity_fakes.idp_description),)
        self.assertCountEqual(datalist, tuple(data))

    def test_identity_provider_list_ID_option(self):
        arglist = ['--id', identity_fakes.idp_id]
        verifylist = [('id', identity_fakes.idp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'id': identity_fakes.idp_id}
        self.identity_providers_mock.list.assert_called_with(**kwargs)
        collist = ('ID', 'Enabled', 'Domain ID', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.idp_id, True, identity_fakes.domain_id, identity_fakes.idp_description),)
        self.assertCountEqual(datalist, tuple(data))

    def test_identity_provider_list_enabled_option(self):
        arglist = ['--enabled']
        verifylist = [('enabled', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True}
        self.identity_providers_mock.list.assert_called_with(**kwargs)
        collist = ('ID', 'Enabled', 'Domain ID', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.idp_id, True, identity_fakes.domain_id, identity_fakes.idp_description),)
        self.assertCountEqual(datalist, tuple(data))