from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_group_type_access as osc_sgta
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupTypeAccessList(TestShareGroupTypeAccess):
    columns = ['Project ID']
    data = (('',), ('',))

    def setUp(self):
        super(TestShareGroupTypeAccessList, self).setUp()
        self.type_access_mock.list.return_value = (self.columns, self.data)
        self.cmd = osc_sgta.ListShareGroupTypeAccess(self.app, None)

    def test_share_group_type_access_list(self):
        share_group_type = manila_fakes.FakeShareGroupType.create_one_share_group_type(attrs={'is_public': False})
        self.share_group_types_mock.get.return_value = share_group_type
        arglist = [share_group_type.id]
        verifylist = [('share_group_type', share_group_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.type_access_mock.list.assert_called_once_with(share_group_type)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_share_group_type_access_list_public_type(self):
        share_group_type = manila_fakes.FakeShareGroupType.create_one_share_group_type(attrs={'is_public': True})
        self.share_group_types_mock.get.return_value = share_group_type
        arglist = [share_group_type.id]
        verifylist = [('share_group_type', share_group_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)