import testtools
from unittest import mock
from aodhclient import exceptions
from aodhclient.v2 import quota_cli
class QuotaShowTest(testtools.TestCase):

    def setUp(self):
        super(QuotaShowTest, self).setUp()
        self.app = mock.Mock()
        self.quota_mgr_mock = self.app.client_manager.alarming.quota
        self.parser = mock.Mock()
        self.quota_show = quota_cli.QuotaShow(self.app, self.parser)

    def test_quota_show(self):
        self.quota_mgr_mock.list.return_value = {'project_id': 'fake_project', 'quotas': [{'limit': 20, 'resource': 'alarms'}]}
        parser = self.quota_show.get_parser('')
        args = parser.parse_args(['--project', 'fake_project'])
        ret = list(self.quota_show.take_action(args))
        self.quota_mgr_mock.list.assert_called_once_with(project='fake_project')
        self.assertIn('alarms', ret[0])
        self.assertIn(20, ret[1])