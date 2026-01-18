import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestProxyGet(base.TestCase):

    def setUp(self):
        super(TestProxyGet, self).setUp()
        self.session = mock.Mock()
        self.session._sdk_connection = self.cloud
        self.fake_id = 1
        self.fake_name = 'fake_name'
        self.fake_result = 'fake_result'
        self.res = mock.Mock(spec=RetrieveableResource)
        self.res.id = self.fake_id
        self.res.fetch = mock.Mock(return_value=self.fake_result)
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        RetrieveableResource.new = mock.Mock(return_value=self.res)

    def test_get_resource(self):
        rv = self.sot._get(RetrieveableResource, self.res)
        self.res.fetch.assert_called_with(self.sot, requires_id=True, base_path=None, skip_cache=mock.ANY, error_message=mock.ANY)
        self.assertEqual(rv, self.fake_result)

    def test_get_resource_with_args(self):
        args = {'key': 'value'}
        rv = self.sot._get(RetrieveableResource, self.res, **args)
        self.res._update.assert_called_once_with(**args)
        self.res.fetch.assert_called_with(self.sot, requires_id=True, base_path=None, skip_cache=mock.ANY, error_message=mock.ANY)
        self.assertEqual(rv, self.fake_result)

    def test_get_id(self):
        rv = self.sot._get(RetrieveableResource, self.fake_id)
        RetrieveableResource.new.assert_called_with(connection=self.cloud, id=self.fake_id)
        self.res.fetch.assert_called_with(self.sot, requires_id=True, base_path=None, skip_cache=mock.ANY, error_message=mock.ANY)
        self.assertEqual(rv, self.fake_result)

    def test_get_base_path(self):
        base_path = 'dummy'
        rv = self.sot._get(RetrieveableResource, self.fake_id, base_path=base_path)
        RetrieveableResource.new.assert_called_with(connection=self.cloud, id=self.fake_id)
        self.res.fetch.assert_called_with(self.sot, requires_id=True, base_path=base_path, skip_cache=mock.ANY, error_message=mock.ANY)
        self.assertEqual(rv, self.fake_result)

    def test_get_not_found(self):
        self.res.fetch.side_effect = exceptions.ResourceNotFound(message='test', http_status=404)
        self.assertRaisesRegex(exceptions.ResourceNotFound, 'test', self.sot._get, RetrieveableResource, self.res)