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
class AvailableVersionsTests(utils.TestCase):

    def setUp(self):
        super(AvailableVersionsTests, self).setUp()
        self.deprecations.expect_deprecations()

    def test_available_versions_basics(self):
        examples = {'keystone': V3_VERSION_LIST, 'cinder': jsonutils.dumps(CINDER_EXAMPLES), 'glance': jsonutils.dumps(GLANCE_EXAMPLES)}
        for path, text in examples.items():
            url = '%s%s' % (BASE_URL, path)
            self.requests_mock.get(url, status_code=300, text=text)
            versions = discover.available_versions(url)
            for v in versions:
                for n in ('id', 'status', 'links'):
                    msg = '%s missing from %s version data' % (n, path)
                    self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))

    def test_available_versions_individual(self):
        self.requests_mock.get(V3_URL, status_code=200, text=V3_VERSION_ENTRY)
        versions = discover.available_versions(V3_URL)
        for v in versions:
            self.assertEqual(v['id'], 'v3.0')
            self.assertEqual(v['status'], 'stable')
            self.assertIn('media-types', v)
            self.assertIn('links', v)

    def test_available_keystone_data(self):
        self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
        versions = discover.available_versions(BASE_URL)
        self.assertEqual(2, len(versions))
        for v in versions:
            self.assertIn(v['id'], ('v2.0', 'v3.0'))
            self.assertEqual(v['updated'], UPDATED)
            self.assertEqual(v['status'], 'stable')
            if v['id'] == 'v3.0':
                self.assertEqual(v['media-types'], V3_MEDIA_TYPES)

    def test_available_cinder_data(self):
        text = jsonutils.dumps(CINDER_EXAMPLES)
        self.requests_mock.get(BASE_URL, status_code=300, text=text)
        versions = discover.available_versions(BASE_URL)
        self.assertEqual(2, len(versions))
        for v in versions:
            self.assertEqual(v['status'], 'CURRENT')
            if v['id'] == 'v1.0':
                self.assertEqual(v['updated'], '2012-01-04T11:33:21Z')
            elif v['id'] == 'v2.0':
                self.assertEqual(v['updated'], '2012-11-21T11:33:21Z')
            else:
                self.fail('Invalid version found')

    def test_available_glance_data(self):
        text = jsonutils.dumps(GLANCE_EXAMPLES)
        self.requests_mock.get(BASE_URL, status_code=200, text=text)
        versions = discover.available_versions(BASE_URL)
        self.assertEqual(5, len(versions))
        for v in versions:
            if v['id'] in ('v2.2', 'v1.1'):
                self.assertEqual(v['status'], 'CURRENT')
            elif v['id'] in ('v2.1', 'v2.0', 'v1.0'):
                self.assertEqual(v['status'], 'SUPPORTED')
            else:
                self.fail('Invalid version found')