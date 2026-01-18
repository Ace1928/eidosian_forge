from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestQuotaShow(TestQuotas):
    project = identity_fakes.FakeProject.create_one_project()
    user = identity_fakes.FakeUser.create_one_user()

    def setUp(self):
        super(TestQuotaShow, self).setUp()
        self.quotas = manila_fakes.FakeQuotaSet.create_fake_quotas()
        self.quotas_mock.get.return_value = self.quotas
        self.cmd = osc_quotas.QuotaShow(self.app, None)

    def test_quota_show(self):
        arglist = [self.project.id]
        verifylist = [('project', self.project.id)]
        with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
            mock_find_resource.return_value = self.project
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            columns, data = self.cmd.take_action(parsed_args)
            self.quotas_mock.get.assert_called_with(detail=False, tenant_id=self.project.id, user_id=None)
            self.assertCountEqual(columns, self.quotas.keys())
            self.assertCountEqual(data, self.quotas._info.values())

    def test_quota_show_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.38')
        arglist = [self.project.id, '--share-type', 'default']
        verifylist = [('project', self.project.id), ('share_type', 'default')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_quota_show_defaults(self):
        arglist = [self.project.id, '--defaults']
        verifylist = [('project', self.project.id), ('defaults', True)]
        self.quotas_mock.defaults = mock.Mock()
        self.quotas_mock.defaults.return_value = self.quotas
        with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
            mock_find_resource.return_value = self.project
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            columns, data = self.cmd.take_action(parsed_args)
            self.quotas_mock.defaults.assert_called_with(self.project.id)
            self.assertCountEqual(columns, self.quotas.keys())
            self.assertCountEqual(data, self.quotas._info.values())