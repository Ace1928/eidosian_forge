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
class TestProxyBulkCreate(base.TestCase):

    def setUp(self):
        super(TestProxyBulkCreate, self).setUp()

        class Res(resource.Resource):
            pass
        self.session = mock.Mock()
        self.result = mock.sentinel
        self.data = mock.Mock()
        self.sot = proxy.Proxy(self.session)
        self.cls = Res
        self.cls.bulk_create = mock.Mock(return_value=self.result)

    def test_bulk_create_attributes(self):
        rv = self.sot._bulk_create(self.cls, self.data)
        self.assertEqual(rv, self.result)
        self.cls.bulk_create.assert_called_once_with(self.sot, self.data, base_path=None)

    def test_bulk_create_attributes_override_base_path(self):
        base_path = 'dummy'
        rv = self.sot._bulk_create(self.cls, self.data, base_path=base_path)
        self.assertEqual(rv, self.result)
        self.cls.bulk_create.assert_called_once_with(self.sot, self.data, base_path=base_path)