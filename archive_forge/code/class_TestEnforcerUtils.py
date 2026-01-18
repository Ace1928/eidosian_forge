from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
class TestEnforcerUtils(base.BaseTestCase):

    def setUp(self):
        super(TestEnforcerUtils, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='oslo_limit', endpoint_id='ENDPOINT_ID')
        opts.register_opts(CONF)
        self.mock_conn = mock.MagicMock()
        limit._SDK_CONNECTION = self.mock_conn

    def test_get_endpoint(self):
        fake_endpoint = endpoint.Endpoint()
        self.mock_conn.get_endpoint.return_value = fake_endpoint
        utils = limit._EnforcerUtils()
        self.assertEqual(fake_endpoint, utils._endpoint)
        self.mock_conn.get_endpoint.assert_called_once_with('ENDPOINT_ID')

    def test_get_registered_limit_empty(self):
        self.mock_conn.registered_limits.return_value = iter([])
        utils = limit._EnforcerUtils()
        reg_limit = utils._get_registered_limit('foo')
        self.assertIsNone(reg_limit)

    def test_get_registered_limit(self):
        foo = registered_limit.RegisteredLimit()
        foo.resource_name = 'foo'
        self.mock_conn.registered_limits.return_value = iter([foo])
        utils = limit._EnforcerUtils()
        reg_limit = utils._get_registered_limit('foo')
        self.assertEqual(foo, reg_limit)

    def test_get_registered_limits(self):
        fake_endpoint = endpoint.Endpoint()
        fake_endpoint.service_id = 'service_id'
        fake_endpoint.region_id = 'region_id'
        self.mock_conn.get_endpoint.return_value = fake_endpoint
        empty_iterator = iter([])
        a = registered_limit.RegisteredLimit()
        a.resource_name = 'a'
        a.default_limit = 1
        a_iterator = iter([a])
        c = registered_limit.RegisteredLimit()
        c.resource_name = 'c'
        c.default_limit = 2
        c_iterator = iter([c])
        self.mock_conn.registered_limits.side_effect = [a_iterator, empty_iterator, c_iterator]
        utils = limit._EnforcerUtils()
        limits = utils.get_registered_limits(['a', 'b', 'c'])
        self.assertEqual([('a', 1), ('b', 0), ('c', 2)], limits)

    def test_get_project_limits(self):
        fake_endpoint = endpoint.Endpoint()
        fake_endpoint.service_id = 'service_id'
        fake_endpoint.region_id = 'region_id'
        self.mock_conn.get_endpoint.return_value = fake_endpoint
        project_id = uuid.uuid4().hex
        empty_iterator = iter([])
        a = klimit.Limit()
        a.resource_name = 'a'
        a.resource_limit = 1
        a_iterator = iter([a])
        self.mock_conn.limits.side_effect = [a_iterator, empty_iterator, empty_iterator, empty_iterator]
        b = registered_limit.RegisteredLimit()
        b.resource_name = 'b'
        b.default_limit = 2
        b_iterator = iter([b])
        self.mock_conn.registered_limits.side_effect = [b_iterator, empty_iterator, empty_iterator]
        utils = limit._EnforcerUtils()
        limits = utils.get_project_limits(project_id, ['a', 'b'])
        self.assertEqual([('a', 1), ('b', 2)], limits)
        limits = utils.get_project_limits(project_id, ['c', 'd'])
        self.assertEqual([('c', 0), ('d', 0)], limits)

    def test__get_project_limit_cache(self, cache=True):
        project_id = uuid.uuid4().hex
        fix = self.useFixture(fixture.LimitFixture({'foo': 5}, {project_id: {'foo': 3}}))
        utils = limit._EnforcerUtils(cache=cache)
        foo_limit = utils._get_project_limit(project_id, 'foo')
        self.assertEqual(3, foo_limit.resource_limit)
        self.assertEqual(1, fix.mock_conn.limits.call_count)
        foo_limit = utils._get_project_limit(project_id, 'foo')
        count = 1 if cache else 2
        self.assertEqual(count, fix.mock_conn.limits.call_count)

    def test__get_project_limit_cache_no_cache(self):
        self.test__get_project_limit_cache(cache=False)

    def test__get_registered_limit_cache(self, cache=True):
        project_id = uuid.uuid4().hex
        fix = self.useFixture(fixture.LimitFixture({'foo': 5}, {project_id: {'foo': 3}}))
        utils = limit._EnforcerUtils(cache=cache)
        foo_limit = utils._get_registered_limit('foo')
        self.assertEqual(5, foo_limit.default_limit)
        self.assertEqual(1, fix.mock_conn.registered_limits.call_count)
        foo_limit = utils._get_registered_limit('foo')
        count = 1 if cache else 2
        self.assertEqual(count, fix.mock_conn.registered_limits.call_count)

    def test__get_registered_limit_cache_no_cache(self):
        self.test__get_registered_limit_cache(cache=False)

    def test_get_limit_cache(self, cache=True):
        fix = self.useFixture(fixture.LimitFixture({'foo': 5}, {}))
        project_id = uuid.uuid4().hex
        utils = limit._EnforcerUtils(cache=cache)
        foo_limit = utils._get_limit(project_id, 'foo')
        self.assertEqual(5, foo_limit)
        self.assertEqual(1, fix.mock_conn.registered_limits.call_count)
        foo_limit = utils._get_limit(project_id, 'foo')
        self.assertEqual(5, foo_limit)
        count = 1 if cache else 2
        self.assertEqual(count, fix.mock_conn.registered_limits.call_count)
        fix.projlimits[project_id] = {'foo': 1}
        foo_limit = utils._get_limit(project_id, 'foo')
        self.assertEqual(1, foo_limit)
        self.assertEqual(3, fix.mock_conn.limits.call_count)
        foo_limit = utils._get_limit(project_id, 'foo')
        self.assertEqual(1, foo_limit)
        count = 3 if cache else 4
        self.assertEqual(count, fix.mock_conn.limits.call_count)

    def test_get_limit_no_cache(self):
        self.test_get_limit_cache(cache=False)

    def test_get_limit(self):
        utils = limit._EnforcerUtils(cache=False)
        mgpl = mock.MagicMock()
        mgrl = mock.MagicMock()
        with mock.patch.multiple(utils, _get_project_limit=mgpl, _get_registered_limit=mgrl):
            utils._get_limit('project', 'foo')
            mgrl.assert_not_called()
            mgpl.assert_called_once_with('project', 'foo')
            mgrl.reset_mock()
            mgpl.reset_mock()
            mgpl.return_value = None
            utils._get_limit('project', 'foo')
            mgrl.assert_called_once_with('foo')
            mgpl.assert_called_once_with('project', 'foo')
            mgrl.reset_mock()
            mgpl.reset_mock()
            utils._get_limit(None, 'foo')
            mgrl.assert_called_once_with('foo')
            mgpl.assert_not_called()