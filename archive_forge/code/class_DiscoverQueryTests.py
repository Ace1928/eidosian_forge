import re
import uuid
from keystoneauth1 import fixture
from oslo_serialization import jsonutils
from testtools import matchers
from keystoneclient import _discover
from keystoneclient.auth import token_endpoint
from keystoneclient import client
from keystoneclient import discover
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
class DiscoverQueryTests(utils.TestCase):

    def test_available_keystone_data(self):
        self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data()
        self.assertEqual((2, 0), versions[0]['version'])
        self.assertEqual('stable', versions[0]['raw_status'])
        self.assertEqual(V2_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[1]['version'])
        self.assertEqual('stable', versions[1]['raw_status'])
        self.assertEqual(V3_URL, versions[1]['url'])
        version = disc.data_for('v3.0')
        self.assertEqual((3, 0), version['version'])
        self.assertEqual('stable', version['raw_status'])
        self.assertEqual(V3_URL, version['url'])
        version = disc.data_for(2)
        self.assertEqual((2, 0), version['version'])
        self.assertEqual('stable', version['raw_status'])
        self.assertEqual(V2_URL, version['url'])
        self.assertIsNone(disc.url_for('v4'))
        self.assertEqual(V3_URL, disc.url_for('v3'))
        self.assertEqual(V2_URL, disc.url_for('v2'))

    def test_available_cinder_data(self):
        text = jsonutils.dumps(CINDER_EXAMPLES)
        self.requests_mock.get(BASE_URL, status_code=300, text=text)
        v1_url = '%sv1/' % BASE_URL
        v2_url = '%sv2/' % BASE_URL
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data()
        self.assertEqual((1, 0), versions[0]['version'])
        self.assertEqual('CURRENT', versions[0]['raw_status'])
        self.assertEqual(v1_url, versions[0]['url'])
        self.assertEqual((2, 0), versions[1]['version'])
        self.assertEqual('CURRENT', versions[1]['raw_status'])
        self.assertEqual(v2_url, versions[1]['url'])
        version = disc.data_for('v2.0')
        self.assertEqual((2, 0), version['version'])
        self.assertEqual('CURRENT', version['raw_status'])
        self.assertEqual(v2_url, version['url'])
        version = disc.data_for(1)
        self.assertEqual((1, 0), version['version'])
        self.assertEqual('CURRENT', version['raw_status'])
        self.assertEqual(v1_url, version['url'])
        self.assertIsNone(disc.url_for('v3'))
        self.assertEqual(v2_url, disc.url_for('v2'))
        self.assertEqual(v1_url, disc.url_for('v1'))

    def test_available_glance_data(self):
        text = jsonutils.dumps(GLANCE_EXAMPLES)
        self.requests_mock.get(BASE_URL, text=text)
        v1_url = '%sv1/' % BASE_URL
        v2_url = '%sv2/' % BASE_URL
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data()
        self.assertEqual((1, 0), versions[0]['version'])
        self.assertEqual('SUPPORTED', versions[0]['raw_status'])
        self.assertEqual(v1_url, versions[0]['url'])
        self.assertEqual((1, 1), versions[1]['version'])
        self.assertEqual('CURRENT', versions[1]['raw_status'])
        self.assertEqual(v1_url, versions[1]['url'])
        self.assertEqual((2, 0), versions[2]['version'])
        self.assertEqual('SUPPORTED', versions[2]['raw_status'])
        self.assertEqual(v2_url, versions[2]['url'])
        self.assertEqual((2, 1), versions[3]['version'])
        self.assertEqual('SUPPORTED', versions[3]['raw_status'])
        self.assertEqual(v2_url, versions[3]['url'])
        self.assertEqual((2, 2), versions[4]['version'])
        self.assertEqual('CURRENT', versions[4]['raw_status'])
        self.assertEqual(v2_url, versions[4]['url'])
        for ver in (2, 2.1, 2.2):
            version = disc.data_for(ver)
            self.assertEqual((2, 2), version['version'])
            self.assertEqual('CURRENT', version['raw_status'])
            self.assertEqual(v2_url, version['url'])
            self.assertEqual(v2_url, disc.url_for(ver))
        for ver in (1, 1.1):
            version = disc.data_for(ver)
            self.assertEqual((1, 1), version['version'])
            self.assertEqual('CURRENT', version['raw_status'])
            self.assertEqual(v1_url, version['url'])
            self.assertEqual(v1_url, disc.url_for(ver))
        self.assertIsNone(disc.url_for('v3'))
        self.assertIsNone(disc.url_for('v2.3'))

    def test_allow_deprecated(self):
        status = 'deprecated'
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': status, 'updated': UPDATED}]
        text = jsonutils.dumps({'versions': version_list})
        self.requests_mock.get(BASE_URL, text=text)
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data(allow_deprecated=False)
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_deprecated=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_allow_experimental(self):
        status = 'experimental'
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': status, 'updated': UPDATED}]
        text = jsonutils.dumps({'versions': version_list})
        self.requests_mock.get(BASE_URL, text=text)
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data()
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_experimental=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_allow_unknown(self):
        status = 'abcdef'
        version_list = fixture.DiscoveryList(BASE_URL, v2=False, v3_status=status)
        self.requests_mock.get(BASE_URL, json=version_list)
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.version_data()
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_unknown=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_ignoring_invalid_lnks(self):
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'id': 'v3.1', 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED, 'links': [{'href': V3_URL, 'rel': 'self'}]}]
        text = jsonutils.dumps({'versions': version_list})
        self.requests_mock.get(BASE_URL, text=text)
        with self.deprecations.expect_deprecations_here():
            disc = discover.Discover(auth_url=BASE_URL)
        versions = disc.raw_version_data()
        self.assertEqual(3, len(versions))
        versions = disc.version_data()
        self.assertEqual(1, len(versions))