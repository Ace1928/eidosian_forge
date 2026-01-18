from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_group_type_access as osc_sgta
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupTypeAccessAllow(TestShareGroupTypeAccess):

    def setUp(self):
        super(TestShareGroupTypeAccessAllow, self).setUp()
        self.project = identity_fakes.FakeProject.create_one_project()
        self.share_group_type = manila_fakes.FakeShareGroupType.create_one_share_group_type(attrs={'is_public': False})
        self.share_group_types_mock.get.return_value = self.share_group_type
        self.projects_mock.get.return_value = self.project
        self.type_access_mock.add_project_access.return_value = None
        self.cmd = osc_sgta.ShareGroupTypeAccessAllow(self.app, None)

    def test_share_group_type_access_create(self):
        arglist = [self.share_group_type.id, self.project.id]
        verifylist = [('share_group_type', self.share_group_type.id), ('projects', [self.project.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.type_access_mock.add_project_access.assert_called_with(self.share_group_type, self.project.id)
        self.assertIsNone(result)

    def test_share_group_type_access_create_invalid_project_exception(self):
        arglist = [self.share_group_type.id, 'invalid_project_format']
        verifylist = [('share_group_type', self.share_group_type.id), ('projects', ['invalid_project_format'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.type_access_mock.add_project_access.side_effect = BadRequest()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)