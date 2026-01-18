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
class TestProxyDelete(base.TestCase):

    def setUp(self):
        super(TestProxyDelete, self).setUp()
        self.session = mock.Mock()
        self.session._sdk_connection = self.cloud
        self.fake_id = 1
        self.res = mock.Mock(spec=DeleteableResource)
        self.res.id = self.fake_id
        self.res.delete = mock.Mock()
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        DeleteableResource.new = mock.Mock(return_value=self.res)

    def test_delete(self):
        self.sot._delete(DeleteableResource, self.res)
        self.res.delete.assert_called_with(self.sot)
        self.sot._delete(DeleteableResource, self.fake_id)
        DeleteableResource.new.assert_called_with(connection=self.cloud, id=self.fake_id)
        self.res.delete.assert_called_with(self.sot)
        self.res.delete.return_value = self.fake_id
        rv = self.sot._delete(DeleteableResource, self.fake_id)
        self.assertEqual(rv, self.fake_id)

    def test_delete_ignore_missing(self):
        self.res.delete.side_effect = exceptions.ResourceNotFound(message='test', http_status=404)
        rv = self.sot._delete(DeleteableResource, self.fake_id)
        self.assertIsNone(rv)

    def test_delete_NotFound(self):
        self.res.delete.side_effect = exceptions.ResourceNotFound(message='test', http_status=404)
        self.assertRaisesRegex(exceptions.ResourceNotFound, 'test', self.sot._delete, DeleteableResource, self.res, ignore_missing=False)

    def test_delete_HttpException(self):
        self.res.delete.side_effect = exceptions.HttpException(message='test', http_status=500)
        self.assertRaises(exceptions.HttpException, self.sot._delete, DeleteableResource, self.res, ignore_missing=False)