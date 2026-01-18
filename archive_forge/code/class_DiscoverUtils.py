import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class DiscoverUtils(utils.TestCase):

    def test_version_number(self):

        def assertVersion(out, inp):
            self.assertEqual(out, discover.normalize_version_number(inp))

        def versionRaises(inp):
            self.assertRaises(TypeError, discover.normalize_version_number, inp)
        assertVersion((1, 2), 'v1.2')
        assertVersion((11, 0), 'v11')
        assertVersion((1, 2), '1.2')
        assertVersion((1, 5, 1), '1.5.1')
        assertVersion((1, 0), '1')
        assertVersion((1, 0), 1)
        assertVersion((5, 2), 5.2)
        assertVersion((3, 20), '3.20')
        assertVersion((6, 1), (6, 1))
        assertVersion((1, 40), [1, 40])
        assertVersion((1, 0), (1,))
        assertVersion((1, 0), ['1'])
        assertVersion((discover.LATEST, discover.LATEST), 'latest')
        assertVersion((discover.LATEST, discover.LATEST), ['latest'])
        assertVersion((discover.LATEST, discover.LATEST), discover.LATEST)
        assertVersion((discover.LATEST, discover.LATEST), (discover.LATEST,))
        assertVersion((10, discover.LATEST), '10.latest')
        assertVersion((10, discover.LATEST), (10, 'latest'))
        assertVersion((10, discover.LATEST), (10, discover.LATEST))
        versionRaises(None)
        versionRaises('hello')
        versionRaises('1.a')
        versionRaises('vacuum')
        versionRaises('')
        versionRaises(('1', 'a'))

    def test_version_args(self):
        """Validate _normalize_version_args."""

        def assert_min_max(in_ver, in_min, in_max, in_type, out_min, out_max):
            self.assertEqual((out_min, out_max), discover._normalize_version_args(in_ver, in_min, in_max, service_type=in_type))

        def normalize_raises(ver, min, max, in_type):
            self.assertRaises(ValueError, discover._normalize_version_args, ver, min, max, service_type=in_type)
        assert_min_max(None, None, None, None, None, None)
        assert_min_max(None, None, 'v1.2', None, None, (1, 2))
        assert_min_max(None, 'v1.2', 'latest', None, (1, 2), (discover.LATEST, discover.LATEST))
        assert_min_max(None, 'v1.2', '1.6', None, (1, 2), (1, 6))
        assert_min_max(None, 'v1.2', '1.latest', None, (1, 2), (1, discover.LATEST))
        assert_min_max(None, 'latest', 'latest', None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
        assert_min_max(None, 'latest', None, None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
        assert_min_max(None, (1, 2), None, None, (1, 2), (discover.LATEST, discover.LATEST))
        assert_min_max('', ('1', '2'), (1, 6), None, (1, 2), (1, 6))
        assert_min_max(None, ('1', '2'), (1, discover.LATEST), None, (1, 2), (1, discover.LATEST))
        assert_min_max('v1.2', '', None, None, (1, 2), (1, discover.LATEST))
        assert_min_max('1.latest', None, '', None, (1, discover.LATEST), (1, discover.LATEST))
        assert_min_max('v1', None, None, None, (1, 0), (1, discover.LATEST))
        assert_min_max('latest', None, None, None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
        assert_min_max(None, None, 'latest', 'volumev2', (2, 0), (2, discover.LATEST))
        assert_min_max(None, None, None, 'volumev2', (2, 0), (2, discover.LATEST))
        normalize_raises('v1', 'v2', None, None)
        normalize_raises('v1', None, 'v2', None)
        normalize_raises(None, 'latest', 'v1', None)
        normalize_raises(None, 'v1.2', 'v1.1', None)
        normalize_raises(None, 'v1.2', 1, None)
        normalize_raises('v2', None, None, 'volumev3')
        normalize_raises('v3', None, None, 'volumev2')
        normalize_raises(None, 'v2', None, 'volumev3')
        normalize_raises(None, None, 'v3', 'volumev2')

    def test_version_to_string(self):

        def assert_string(out, inp):
            self.assertEqual(out, discover.version_to_string(inp))
        assert_string('latest', (discover.LATEST,))
        assert_string('latest', (discover.LATEST, discover.LATEST))
        assert_string('latest', (discover.LATEST, discover.LATEST, discover.LATEST))
        assert_string('1', (1,))
        assert_string('1.2', (1, 2))
        assert_string('1.latest', (1, discover.LATEST))

    def test_version_between(self):

        def good(minver, maxver, cand):
            self.assertTrue(discover.version_between(minver, maxver, cand))

        def bad(minver, maxver, cand):
            self.assertFalse(discover.version_between(minver, maxver, cand))

        def exc(excls, minver, maxver, cand):
            self.assertRaises(excls, discover.version_between, minver, maxver, cand)
        exc(ValueError, (1, 0), (1, 0), None)
        exc(ValueError, 'v1.0', '1.0', '')
        exc(TypeError, None, None, 'bogus')
        exc(TypeError, None, None, (1, 'two'))
        exc(TypeError, 'bogus', None, (1, 0))
        exc(TypeError, (1, 'two'), None, (1, 0))
        exc(TypeError, None, 'bogus', (1, 0))
        exc(TypeError, None, (1, 'two'), (1, 0))
        bad((2, 4), None, (1, 55))
        bad('v2.4', '', '2.3')
        bad('latest', None, (2, 3000))
        bad((2, discover.LATEST), '', 'v2.3000')
        bad((2, 3000), '', (1, discover.LATEST))
        bad('latest', None, 'v1000.latest')
        bad(None, (2, 4), (2, 5))
        bad('', 'v2.4', '2.5')
        bad(None, (2, discover.LATEST), (3, 0))
        bad('', '2000.latest', 'latest')
        good((1, 0), (1, 0), (1, 0))
        good('1.0', '2.9', '1.0')
        good('v1.0', 'v2.9', 'v2.9')
        good((1, 0), (1, 10), (1, 2))
        good('1', '2', '1.2')
        good(None, (2, 5), (2, 3))
        good('2.5', '', '2.6')
        good('', '', 'v1')
        good(None, None, (999, 999))
        good(None, None, 'latest')
        good((discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
        good((discover.LATEST, discover.LATEST), None, (discover.LATEST, discover.LATEST))
        good('', 'latest', 'latest')
        good('2.latest', '3.latest', '3.0')
        good('2.latest', None, (55, 66))
        good(None, '3.latest', '3.9999')