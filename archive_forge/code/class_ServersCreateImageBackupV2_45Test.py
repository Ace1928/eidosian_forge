import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
class ServersCreateImageBackupV2_45Test(utils.FixturedTestCase):
    """Tests the 2.45 microversion for createImage and createBackup
    server actions.
    """
    client_fixture_class = client.V1
    data_fixture_class = data.V1
    api_version = '2.45'

    def setUp(self):
        super(ServersCreateImageBackupV2_45Test, self).setUp()
        self.cs.api_version = api_versions.APIVersion(self.api_version)

    def test_create_image(self):
        """Tests the createImage API with the 2.45 microversion which
        does not return the Location header, it returns a json dict in the
        response body with an image_id key.
        """
        s = self.cs.servers.get(1234)
        im = s.create_image('123')
        self.assertEqual('456', im)
        self.assert_request_id(im, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')
        im = s.create_image('123', {})
        self.assert_request_id(im, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')
        im = self.cs.servers.create_image(s, '123')
        self.assert_request_id(im, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')
        im = self.cs.servers.create_image(s, '123', {})
        self.assert_request_id(im, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')

    def test_backup(self):
        s = self.cs.servers.get(1234)
        sb = s.backup('back1', 'daily', 1)
        self.assertIn('image_id', sb)
        self.assertEqual('456', sb['image_id'])
        self.assert_request_id(sb, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')
        sb = self.cs.servers.backup(s, 'back1', 'daily', 2)
        self.assertEqual('456', sb['image_id'])
        self.assert_request_id(sb, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action')