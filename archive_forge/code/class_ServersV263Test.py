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
class ServersV263Test(ServersV257Test):
    api_version = '2.63'

    def test_create_server_with_trusted_image_certificates(self):
        self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics(), trusted_image_certificates=['id1', 'id2'])
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'key_name': 'fakekey', 'max_count': 1, 'metadata': {'foo': 'bar'}, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'trusted_image_certificates': ['id1', 'id2'], 'user_data': 'aGVsbG8gbW90bw=='}})

    def test_create_server_with_trusted_image_certificates_pre_263_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.62')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics(), trusted_image_certificates=['id1', 'id2'])
        self.assertIn('trusted_image_certificates', str(ex))

    def test_rebuild_server_with_trusted_image_certificates(self):
        s = self.cs.servers.get(1234)
        ret = s.rebuild(image='1', trusted_image_certificates=['id1', 'id2'])
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'rebuild': {'imageRef': '1', 'trusted_image_certificates': ['id1', 'id2']}})

    def test_rebuild_server_with_trusted_image_certificates_none(self):
        s = self.cs.servers.get(1234)
        ret = s.rebuild(image='1', trusted_image_certificates=None)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'rebuild': {'imageRef': '1', 'trusted_image_certificates': None}})

    def test_rebuild_with_trusted_image_certificates_pre_263_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.62')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.rebuild, '1234', fakes.FAKE_IMAGE_UUID_1, trusted_image_certificates=['id1', 'id2'])
        self.assertIn('trusted_image_certificates', str(ex))