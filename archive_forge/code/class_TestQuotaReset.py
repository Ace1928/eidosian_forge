import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestQuotaReset(TestQuota):

    def setUp(self):
        super().setUp()
        self.cmd = quota.ResetQuota(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_quota_attrs')
    def test_quota_reset(self, mock_attrs):
        project_id = 'fake_project_id'
        attrs = {'project_id': project_id}
        qt_reset = fakes.createFakeResource('quota', attrs)
        mock_attrs.return_value = qt_reset.to_dict()
        arglist = [project_id]
        verifylist = [('project', project_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.quota_reset.assert_called_with(project_id=qt_reset.project_id)