from unittest import mock
from blazarclient.osc import plugin
from blazarclient import tests
class ReservationPluginTests(tests.TestCase):

    @mock.patch('blazarclient.v1.client.Client')
    def test_make_client(self, mock_client):
        instance = mock.Mock()
        instance._api_version = {'reservation': '1'}
        endpoint = 'blazar_endpoint'
        instance.get_endpoint_for_service_type = mock.Mock(return_value=endpoint)
        plugin.make_client(instance)
        mock_client.assert_called_with('1', session=instance.session, endpoint_override=endpoint)