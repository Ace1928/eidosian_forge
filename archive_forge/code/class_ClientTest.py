import testtools
from unittest import mock
from magnumclient import client
class ClientTest(testtools.TestCase):

    @mock.patch('magnumclient.v1.client.Client')
    def test_no_version_argument(self, mock_magnum_client):
        client.Client(auth_token='mytoken', magnum_url='http://myurl/')
        mock_magnum_client.assert_called_with(auth_token='mytoken', magnum_url='http://myurl/')

    @mock.patch('magnumclient.v1.client.Client')
    def test_valid_version_argument(self, mock_magnum_client):
        client.Client(version='1', magnum_url='http://myurl/')
        mock_magnum_client.assert_called_with(magnum_url='http://myurl/')

    @mock.patch('magnumclient.v1.client.Client')
    def test_invalid_version_argument(self, mock_magnum_client):
        self.assertRaises(ValueError, client.Client, version='2', magnum_url='http://myurl/')