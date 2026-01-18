from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
class SenlinClientPluginTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(SenlinClientPluginTest, self).setUp()
        context = utils.dummy_context()
        self.plugin = context.clients.client_plugin('senlin')
        self.client = self.plugin.client()

    def test_cluster_get(self):
        self.assertIsNotNone(self.client.clusters)

    def test_is_bad_request(self):
        self.assertTrue(self.plugin.is_bad_request(exceptions.HttpException(http_status=400)))
        self.assertFalse(self.plugin.is_bad_request(Exception))
        self.assertFalse(self.plugin.is_bad_request(exceptions.HttpException(http_status=404)))

    def test_check_action_success(self):
        mock_action = mock.MagicMock()
        mock_action.status = 'SUCCEEDED'
        mock_get = self.patchobject(self.client, 'get_action')
        mock_get.return_value = mock_action
        self.assertTrue(self.plugin.check_action_status('fake_id'))
        mock_get.assert_called_once_with('fake_id')

    def test_get_profile_id(self):
        mock_profile = mock.Mock(id='fake_profile_id')
        mock_get = self.patchobject(self.client, 'get_profile', return_value=mock_profile)
        ret = self.plugin.get_profile_id('fake_profile')
        self.assertEqual('fake_profile_id', ret)
        mock_get.assert_called_once_with('fake_profile')

    def test_get_cluster_id(self):
        mock_cluster = mock.Mock(id='fake_cluster_id')
        mock_get = self.patchobject(self.client, 'get_cluster', return_value=mock_cluster)
        ret = self.plugin.get_cluster_id('fake_cluster')
        self.assertEqual('fake_cluster_id', ret)
        mock_get.assert_called_once_with('fake_cluster')

    def test_get_policy_id(self):
        mock_policy = mock.Mock(id='fake_policy_id')
        mock_get = self.patchobject(self.client, 'get_policy', return_value=mock_policy)
        ret = self.plugin.get_policy_id('fake_policy')
        self.assertEqual('fake_policy_id', ret)
        mock_get.assert_called_once_with('fake_policy')